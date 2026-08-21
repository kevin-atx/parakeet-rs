use std::path::{Path, PathBuf};
use std::{fmt, rc::Rc};

use crate::error::Result;
use ort::session::builder::SessionBuilder;
use ort::session::Session;

// Hardware acceleration options. CPU is default and most reliable.
// GPU providers (CUDA, TensorRT, MIGraphX) offer 5-10x speedup but require specific hardware.
// All GPU providers automatically fall back to CPU if they fail.
//
// Note: CoreML EP currently runs slower than CPU for Sortformer/Parakeet models because
// the ONNX graphs have dynamic input shapes, preventing CoreML from building optimised
// execution plans for ANE/GPU. CoreML claims nodes but runs them on CPU with overhead.
//
// WebGPU is experimental and may produce incorrect results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "tensorrt")]
    TensorRT,
    #[cfg(feature = "coreml")]
    CoreML,
    #[cfg(feature = "directml")]
    DirectML,
    #[cfg(feature = "migraphx")]
    MIGraphX,
    #[cfg(feature = "openvino")]
    OpenVINO,
    #[cfg(feature = "webgpu")]
    WebGPU,
    #[cfg(feature = "nnapi")]
    NNAPI,
}

/// Which compute units the CoreML execution provider may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoreMLComputeUnits {
    All,
    CpuAndNeuralEngine,
    #[default]
    CpuAndGpu,
    CpuOnly,
}

#[derive(Clone)]
pub struct ModelConfig {
    pub execution_provider: ExecutionProvider,
    pub intra_threads: usize,
    pub inter_threads: usize,
    pub configure: Option<Rc<dyn Fn(SessionBuilder) -> ort::Result<SessionBuilder>>>,
    /// Optional cache directory for compiled CoreML models. When set, avoids
    /// recompiling the ONNX-to-CoreML conversion on each session load (~5s).
    /// Only used when execution_provider is CoreML.
    pub coreml_cache_dir: Option<PathBuf>,
    pub coreml_compute_units: CoreMLComputeUnits,
    /// Whether the CPU execution provider uses ORT's BFC arena allocator.
    /// Defaults to true (ORT's own default). Set FALSE for sessions whose
    /// activation shapes VARY RUN TO RUN over a long-lived process: the BFC
    /// arena grows in 128 MB extents, never returns memory to the OS, and
    /// fragments under varying large requests — a daemon hosting the
    /// streaming Sortformer accumulated ~950 such extents (118 GB virtual,
    /// 21 GB swapped) over one 14-hour day (recogment, 2026-08-21; the
    /// allocation stacks name BFCArena::Extend under Sortformer::streaming_update).
    /// Note the GPU-fallback arms already register ort's `CPU` EP with its
    /// derived default `use_arena = false`, so those paths were never affected;
    /// only the pure-Cpu arm inherited ORT's arena-on default.
    pub cpu_arena: bool,
}

impl fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelConfig")
            .field("execution_provider", &self.execution_provider)
            .field("intra_threads", &self.intra_threads)
            .field("inter_threads", &self.inter_threads)
            .field(
                "configure",
                &if self.configure.is_some() {
                    "<fn>"
                } else {
                    "None"
                },
            )
            .field("coreml_cache_dir", &self.coreml_cache_dir)
            .field("coreml_compute_units", &self.coreml_compute_units)
            .field("cpu_arena", &self.cpu_arena)
            .finish()
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            intra_threads: 4,
            inter_threads: 1,
            configure: None,
            coreml_cache_dir: None,
            coreml_compute_units: CoreMLComputeUnits::default(),
            cpu_arena: true,
        }
    }
}

impl ModelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_provider = provider;
        self
    }

    /// See [`ModelConfig::cpu_arena`].
    pub fn with_cpu_arena(mut self, enable: bool) -> Self {
        self.cpu_arena = enable;
        self
    }

    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = threads;
        self
    }

    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = threads;
        self
    }

    pub fn with_custom_configure(
        mut self,
        configure: impl Fn(SessionBuilder) -> ort::Result<SessionBuilder> + 'static,
    ) -> Self {
        self.configure = Some(Rc::new(configure));
        self
    }

    /// Set cache directory for compiled CoreML models.
    /// Avoids ~5s recompilation on each session load.
    pub fn with_coreml_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.coreml_cache_dir = Some(path.into());
        self
    }

    /// Select which compute units the CoreML provider may use.
    /// Defaults to [`CoreMLComputeUnits::CpuAndGpu`];
    pub fn with_coreml_compute_units(mut self, units: CoreMLComputeUnits) -> Self {
        self.coreml_compute_units = units;
        self
    }
    pub(crate) fn build_session(&self, path: &Path) -> Result<Session> {
        let builder = Session::builder()?;
        let mut builder = self.apply_to_session_builder(builder)?;
        Ok(builder.commit_from_file(path)?)
    }

    pub(crate) fn apply_to_session_builder(
        &self,
        builder: SessionBuilder,
    ) -> Result<SessionBuilder> {
        #[cfg(any(
            feature = "cuda",
            feature = "tensorrt",
            feature = "coreml",
            feature = "directml",
            feature = "migraphx",
            feature = "openvino",
            feature = "webgpu",
            feature = "nnapi"
        ))]
        use ort::ep::CPU as CPUExecutionProvider;
        use ort::session::builder::GraphOptimizationLevel;

        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.intra_threads)?
            .with_inter_threads(self.inter_threads)?;

        builder = match self.execution_provider {
            ExecutionProvider::Cpu => {
                if self.cpu_arena {
                    builder
                } else {
                    // Two session options, both required for a dynamic-shape
                    // long-lived session (validated 2026-08-21 on the boundary
                    // reproducer — the arena flag ALONE changed nothing):
                    //  - DisableCpuMemArena: activations come from plain malloc
                    //    and return to the OS on free;
                    //  - memory_pattern(false): ORT otherwise caches a
                    //    peak-sized pre-planned buffer PER DISTINCT INPUT SHAPE
                    //    on the session (ExecutionFrame allocates it whole),
                    //    which is unbounded when shapes vary run to run — the
                    //    actual mechanism behind the 128 MB-per-region growth.
                    builder
                        .with_memory_pattern(false)?
                        .with_execution_providers([
                            ort::ep::CPU::default().with_arena_allocator(false).build(),
                        ])?
                }
            }

            #[cfg(feature = "cuda")]
            ExecutionProvider::Cuda => builder.with_execution_providers([
                ort::ep::CUDA::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "tensorrt")]
            ExecutionProvider::TensorRT => builder.with_execution_providers([
                ort::ep::TensorRT::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "coreml")]
            ExecutionProvider::CoreML => {
                use ort::ep::coreml::{ComputeUnits, CoreML, ModelFormat};
                let units = match self.coreml_compute_units {
                    CoreMLComputeUnits::All => ComputeUnits::All,
                    CoreMLComputeUnits::CpuAndNeuralEngine => ComputeUnits::CPUAndNeuralEngine,
                    CoreMLComputeUnits::CpuAndGpu => ComputeUnits::CPUAndGPU,
                    CoreMLComputeUnits::CpuOnly => ComputeUnits::CPUOnly,
                };
                // MLProgram, not ORT's default NeuralNetwork.
                //
                // The legacy NeuralNetwork format (CoreML 3) cannot convert
                // this model: every chunk fails at
                //   Where node '/encoder/layers.10/self_attn/Where_1'
                //   Status Message: GetElementType is not implemented
                // Measured 2026-07-21 on macOS 26.5 / M1 Pro: 0 chunks
                // transcribed and ~11,900 consecutive failures across both
                // CoreMLComputeUnits::All and CPUAndNeuralEngine, while the
                // CPU EP handled the same audio with zero failures. So CoreML
                // was not slow — it was entirely non-functional.
                //
                // MLProgram (CoreML 5 / macOS 12+) supports a wider operator
                // and type set, which is the documented remedy for exactly
                // this class of conversion gap.
                let mut coreml = CoreML::default()
                    .with_compute_units(units)
                    .with_model_format(ModelFormat::MLProgram);

                if let Some(cache_dir) = &self.coreml_cache_dir {
                    coreml = coreml.with_model_cache_dir(cache_dir.to_string_lossy());
                }

                builder.with_execution_providers([
                    coreml.build(),
                    CPUExecutionProvider::default().build().error_on_failure(),
                ])?
            }

            #[cfg(feature = "directml")]
            ExecutionProvider::DirectML => builder.with_execution_providers([
                ort::ep::DirectML::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "migraphx")]
            ExecutionProvider::MIGraphX => builder.with_execution_providers([
                ort::ep::MIGraphX::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "openvino")]
            ExecutionProvider::OpenVINO => builder.with_execution_providers([
                ort::ep::OpenVINO::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "webgpu")]
            ExecutionProvider::WebGPU => builder.with_execution_providers([
                ort::ep::WebGPU::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "nnapi")]
            ExecutionProvider::NNAPI => builder.with_execution_providers([
                ort::ep::NNAPI::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,
        };

        if let Some(configure) = self.configure.as_ref() {
            builder = configure(builder)?;
        }

        Ok(builder)
    }
}
