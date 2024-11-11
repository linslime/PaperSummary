we model styles as a latent random variable to generate
the most suitable style for the text without requiring reference speech, achieving
efficient latent diffusion.

. We introduce ED-TTS, a multi-scale emo
tional speech synthesis model that leverages Speech Emotion
 Diarization (SED) and Speech Emotion Recognition (SER) to
 model emotions at different levels.

 Our proposed approach yields im
proved performance in both objective and subjective evaluations,
 demonstrating the ability to generate cross-lingual speech with di
verse emotions, even from a neutral source speaker, while preserving
 the speaker’s identity.

Attention-based speech synthesis methods often suffer from
 dispersed attention across the entire input sequence, resulting
 in poor local modeling and unnatural Mandarin synthesized
 speech. To address these issues, we present FastMandarin, a
 rapid and natural Mandarin speech synthesis framework that
 employs two explicit methods to enhance local modeling and
 improve pronunciation representation.

It remains a challenge to effectively control the emotion rendering
 in text-to-speech (TTS) synthesis. Prior studies have primarily fo
cused on learning a global prosodic representation at the utterance
 level, which strongly correlates with linguistic prosody. Our goal
 is to construct a hierarchical emotion distribution (ED) that effec
tively encapsulates intensity variations of emotions at various levels
 of granularity, encompassing phonemes, words, and utterances. Dur
ing TTS training, the hierarchical ED is extracted from the ground
truth audio and guides the predictor to establish a connection be
tween emotional and linguistic prosody. 

The expressive quality of synthesized speech for audiobooks is lim
ited by generalized model architecture and unbalanced style dis
tribution in the training data. 

The expressive quality of synthesized speech for audiobooks is lim
ited by generalized model architecture and unbalanced style dis
tribution in the training data.

It remains a challenge to effectively control the emotion rendering in text-to-speech (TTS) synthesis. Prior studies have primarily focused on learning a global prosodic representation at the utterance level, which strongly correlates with linguistic prosody. This paper introduces a novel non-autoregressive framework that model styles as a latent random variable to generate the most suitable style for the text without requiring reference speech, achieving efficient latent diffusion. Firstly,  We propose a multi-periodic style feature extractor to captures the latent features of different periodic signals in audio. Secondly,  a novel architecture with the multi-periodic style extractor is specially designed to model the pronunciation and high-level style expressiveness respectively。Our proposed approach yields improved performance in both objective and subjective evaluations, demonstrating the ability to generate cross-lingual speech.