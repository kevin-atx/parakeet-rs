use crate::decoder::TimedToken;

/// Timestamp output mode for transcription results
///
/// Determines how token-level timestamps are grouped and presented:
/// - `Tokens`: Raw token-level output from the model (most detailed)
/// - `Words`: Tokens grouped into individual words
/// - `Sentences`: Tokens grouped by sentence boundaries (., ?, !)
///
/// # Model-Specific Recommendations
///
/// - **Parakeet CTC (English)**: Use `Words` mode. The CTC model only outputs lowercase
///   alphabet without punctuation, so sentence segmentation is not possible.
/// - **Parakeet TDT (Multilingual)**: Use `Sentences` mode. The TDT model predicts
///   punctuation, enabling natural sentence boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimestampMode {
    /// Raw token-level timestamps from the model
    #[default]
    Tokens,
    /// Word-level timestamps (groups subword tokens)
    Words,
    /// Sentence-level timestamps (groups by punctuation)
    ///
    /// Note: Only works with models that predict punctuation (e.g., Parakeet TDT).
    /// CTC models don't predict punctuation, so use `Words` mode instead.
    Sentences,
}

/// Convert token timestamps to the requested output mode
///
/// Takes raw token-level timestamps from the model and optionally groups them
/// into words or sentences while preserving the original timing information.
///
/// # Arguments
///
/// * `tokens` - Raw token-level timestamps from model output
/// * `mode` - Desired grouping level (Tokens, Words, or Sentences)
///
/// # Returns
///
/// Vector of TimedToken with timestamps at the requested granularity
pub fn process_timestamps(tokens: &[TimedToken], mode: TimestampMode) -> Vec<TimedToken> {
    match mode {
        TimestampMode::Tokens => tokens.to_vec(),
        TimestampMode::Words => group_by_words(tokens),
        TimestampMode::Sentences => group_by_sentences(tokens),
    }
}

/// Rebuild a flat transcript string from mode-grouped tokens.
///
/// Shared by every timestamped transcribe path so spacing/punctuation rules
/// live in one place:
/// - `Tokens`: concatenate raw SentencePiece pieces and trim.
/// - `Words`: join words with single spaces, attaching standalone punctuation
///   (".", ",", etc.) to the preceding word with no leading space.
/// - `Sentences`: join sentence strings with single spaces.
///
/// Expects `tokens` to already be at the requested granularity (i.e. the output
/// of [`process_timestamps`] for the same `mode`).
pub fn rebuild_text(tokens: &[TimedToken], mode: TimestampMode) -> String {
    match mode {
        TimestampMode::Tokens => tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<String>()
            .trim()
            .to_string(),
        TimestampMode::Words => {
            let mut out = String::new();
            for (i, word) in tokens.iter().map(|t| t.text.as_str()).enumerate() {
                let is_standalone_punct = word.len() == 1
                    && word
                        .chars()
                        .all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'));
                if i > 0 && !is_standalone_punct {
                    out.push(' ');
                }
                out.push_str(word);
            }
            out
        }
        TimestampMode::Sentences => tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    }
}

/// Whether a token begins a word: it carries a SentencePiece word-boundary
/// marker (▁ or a leading space) or is punctuation standing alone — but never a
/// contraction ("'s", "'re") or a hyphenation ("-two"), which attach to the
/// word before them. THE rule; [`group_by_words`] and
/// [`continues_previous_word`] both ask it.
pub(crate) fn token_starts_word(text: &str) -> bool {
    let is_pure_punctuation = !text.is_empty() && text.chars().all(|c| c.is_ascii_punctuation());
    let without_marker = text.trim_start_matches('▁').trim_start_matches(' ');
    let is_contraction = without_marker.starts_with('\'');
    let is_hyphenation = without_marker.starts_with('-');
    (text.starts_with('▁') || text.starts_with(' ') || is_pure_punctuation) && !is_contraction && !is_hyphenation
}

/// Whether the first word [`group_by_words`] makes of `tokens` is the rest of a
/// word begun before them: their first token does not start a word. A stream
/// decoded a call at a time needs this, because `group_by_words` has to start
/// its first word somewhere and cannot see the call before — so a word whose
/// pieces straddle two calls came out as two words ("attitud" | "e").
pub(crate) fn continues_previous_word(tokens: &[TimedToken]) -> bool {
    match tokens.first() {
        Some(t) if !t.text.trim().is_empty() => !token_starts_word(&t.text),
        _ => false,
    }
}

// Group tokens into words based on word boundary markers
pub(crate) fn group_by_words(tokens: &[TimedToken]) -> Vec<TimedToken> {
    let Some(last_token) = tokens.last() else {
        return Vec::new();
    };

    let mut words = Vec::new();
    let mut current_word_text = String::new();
    let mut current_word_start = 0.0;
    let mut current_word_min_conf = 1.0f32;
    let mut last_word_lower = String::new();

    for (i, token) in tokens.iter().enumerate() {
        // Space-only tokens (from SentencePiece ▁ word boundaries) act as word separators
        // but don't contribute text. Save current word if we hit one.
        if token.text.trim().is_empty() {
            if !current_word_text.is_empty() {
                let word_lower = current_word_text.to_lowercase();
                if word_lower != last_word_lower {
                    words.push(TimedToken {
                        text: current_word_text.clone(),
                        start: current_word_start,
                        end: if i > 0 { tokens[i - 1].end } else { token.end },
                        confidence: current_word_min_conf,
                    });
                    last_word_lower = word_lower;
                }
                current_word_text.clear();
                current_word_min_conf = 1.0;
            }
            continue;
        }

        // A new word starts at a word-boundary marker or standalone
        // punctuation, never at a contraction or hyphenation suffix
        // (`token_starts_word`) — and at the first token, which has nothing
        // before it here (`continues_previous_word` says when it should have).
        let starts_word = token_starts_word(&token.text) || i == 0;

        if starts_word && !current_word_text.is_empty() {
            // Save previous word (with deduplication)
            let word_lower = current_word_text.to_lowercase();
            if word_lower != last_word_lower {
                words.push(TimedToken {
                    text: current_word_text.clone(),
                    start: current_word_start,
                    end: tokens[i - 1].end,
                    confidence: current_word_min_conf,
                });
                last_word_lower = word_lower;
            }
            current_word_text.clear();
            current_word_min_conf = 1.0;
        }

        // Start new word or append to current
        if current_word_text.is_empty() {
            current_word_start = token.start;
        }

        // Track minimum confidence across subword tokens in this word.
        current_word_min_conf = current_word_min_conf.min(token.confidence);

        // Add token text, removing word boundary markers
        let token_text = token.text.trim_start_matches('▁').trim_start_matches(' ');
        current_word_text.push_str(token_text);
    }

    // Add final word
    if !current_word_text.is_empty() {
        let word_lower = current_word_text.to_lowercase();
        if word_lower != last_word_lower {
            words.push(TimedToken {
                text: current_word_text,
                start: current_word_start,
                end: last_token.end,
                confidence: current_word_min_conf,
            });
        }
    }

    words
}

// Group words into sentences based on punctuation
fn group_by_sentences(tokens: &[TimedToken]) -> Vec<TimedToken> {
    // First get word-level grouping
    let words = group_by_words(tokens);
    if words.is_empty() {
        return Vec::new();
    }

    let mut sentences = Vec::new();
    let mut current_sentence = Vec::new();

    for word in words {
        // Check if word ends with sentence terminator
        let ends_sentence =
            word.text.contains('.') || word.text.contains('?') || word.text.contains('!');

        current_sentence.push(word);

        if ends_sentence {
            push_sentence(&mut sentences, &current_sentence);
            current_sentence.clear();
        }
    }

    // Add final sentence if exists
    push_sentence(&mut sentences, &current_sentence);

    sentences
}

// Append `words` to `sentences` as a single formatted sentence
fn push_sentence(sentences: &mut Vec<TimedToken>, words: &[TimedToken]) {
    let (Some(first), Some(last)) = (words.first(), words.last()) else {
        return;
    };

    let text = format_sentence(words);
    if !text.is_empty() {
        let confidence = words.iter().map(|w| w.confidence).fold(1.0f32, f32::min);
        sentences.push(TimedToken {
            text,
            start: first.start,
            end: last.end,
            confidence,
        });
    }
}

// Join words with punctuation spacing
fn format_sentence(words: &[TimedToken]) -> String {
    let result: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();

    // Join words, but don't add space before certain punctuation
    let mut output = String::new();
    for (i, word) in result.iter().enumerate() {
        // Check if this word is standalone punctuation that shouldn't have space before it
        // Contractions like "'re" or "'s" should have spaces before them
        let is_standalone_punct = word.len() == 1
            && word
                .chars()
                .all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'));

        if i > 0 && !is_standalone_punct {
            output.push(' ');
        }
        output.push_str(word);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str, start: f32) -> TimedToken {
        TimedToken { text: text.to_string(), start, end: start + 0.08, confidence: 1.0 }
    }

    /// The pieces a call boundary separates, as the 2026-10-18 webinar decoded
    /// them: each is the rest of the word before, and only a token that starts
    /// a word (a marker, or standalone punctuation) begins one.
    #[test]
    fn a_call_that_begins_inside_a_word_says_so() {
        for piece in ["e", "itudes", "'s", "-two"] {
            assert!(continues_previous_word(&[tok(piece, 0.0), tok("▁so", 0.1)]), "{piece:?}");
        }
        for start in ["▁So", " So", ".", ","] {
            assert!(!continues_previous_word(&[tok(start, 0.0)]), "{start:?}");
        }
        assert!(!continues_previous_word(&[]), "no tokens, nothing to continue");
        assert!(!continues_previous_word(&[tok("▁", 0.0), tok("itudes", 0.1)]), "a separator first means a new word");
    }

    /// Lifting the rule into `token_starts_word` must not move a word boundary.
    #[test]
    fn the_shared_rule_groups_words_as_before() {
        let words = group_by_words(&[tok("att", 0.0), tok("itudes", 0.1), tok("▁it", 0.2), tok("'s", 0.3), tok(".", 0.4), tok("▁twenty", 0.5), tok("-two", 0.6)]);
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["attitudes", "it's", ".", "twenty-two"]);
    }

    #[test]
    fn test_word_grouping() {
        let tokens = vec![
            TimedToken {
                text: "▁Hello".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 1.0,
            },
            TimedToken {
                text: "▁world".to_string(),
                start: 0.5,
                end: 1.0,
                confidence: 1.0,
            },
        ];

        let words = group_by_words(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn test_word_grouping_with_hyphenated_word() {
        let tokens = vec![
            TimedToken {
                text: "▁twenty".to_string(),
                start: 0.0,
                end: 0.3,
                confidence: 1.0,
            },
            TimedToken {
                text: "-two".to_string(),
                start: 0.3,
                end: 0.6,
                confidence: 1.0,
            },
            TimedToken {
                text: "▁apples".to_string(),
                start: 0.6,
                end: 1.0,
                confidence: 1.0,
            },
        ];

        let words = group_by_words(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "twenty-two");
        assert_eq!(words[1].text, "apples");
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[0].end, 0.6);
        assert_eq!(words[1].start, 0.6);
        assert_eq!(words[1].end, 1.0);
    }

    #[test]
    fn test_sentence_grouping() {
        let tokens = vec![
            TimedToken {
                text: "▁Hello".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 1.0,
            },
            TimedToken {
                text: "▁world".to_string(),
                start: 0.5,
                end: 1.0,
                confidence: 1.0,
            },
            TimedToken {
                text: ".".to_string(),
                start: 1.0,
                end: 1.1,
                confidence: 1.0,
            },
        ];

        let sentences = group_by_sentences(&tokens);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].text, "Hello world.");
        assert_eq!(sentences[0].start, 0.0);
        assert_eq!(sentences[0].end, 1.1);
    }

    #[test]
    fn test_repetition_preservation() {
        let words = vec![
            TimedToken {
                text: "uh".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 1.0,
            },
            TimedToken {
                text: "uh".to_string(),
                start: 0.5,
                end: 1.0,
                confidence: 1.0,
            },
            TimedToken {
                text: "hello".to_string(),
                start: 1.0,
                end: 1.5,
                confidence: 1.0,
            },
        ];

        let result = format_sentence(&words);
        assert_eq!(result, "uh uh hello");
    }

    #[test]
    fn test_space_token_separates_words_from_digits() {
        // Simulates "like 100" tokenized as [" like", " ", "1", "0", "0"]
        // The space-only token should act as word boundary
        let tokens = vec![
            TimedToken {
                text: " like".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 1.0,
            },
            TimedToken {
                text: " ".to_string(), // Space-only token from ▁
                start: 0.5,
                end: 0.5,
                confidence: 1.0,
            },
            TimedToken {
                text: "1".to_string(),
                start: 0.5,
                end: 0.6,
                confidence: 1.0,
            },
            TimedToken {
                text: "0".to_string(),
                start: 0.6,
                end: 0.7,
                confidence: 1.0,
            },
            TimedToken {
                text: "0".to_string(),
                start: 0.7,
                end: 0.8,
                confidence: 1.0,
            },
        ];

        let words = group_by_words(&tokens);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "like");
        assert_eq!(words[1].text, "100");

        // Also test sentence formatting
        let sentence = format_sentence(&words);
        assert_eq!(sentence, "like 100");
    }
}
