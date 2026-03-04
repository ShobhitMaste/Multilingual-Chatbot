"""
Language detection and EN↔HI translation module.
Uses langdetect for language identification and googletrans for translation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Translator:
    """Handles language detection and English↔Hindi translation."""

    def __init__(self):
        from langdetect import DetectorFactory
        # Set seed for deterministic results
        DetectorFactory.seed = 0
        self._translator = None
        print("✅ Translator initialized (langdetect + googletrans)")

    @property
    def google_translator(self):
        """Lazy-load googletrans translator."""
        if self._translator is None:
            from googletrans import Translator as GoogleTranslator
            self._translator = GoogleTranslator()
        return self._translator

    def detect_language(self, text):
        """
        Detect whether text is Hindi or English.
        
        Args:
            text: Input text string
            
        Returns:
            'hi' for Hindi, 'en' for English
        """
        from langdetect import detect

        if not text or len(text.strip()) < 2:
            return 'en'  # Default to English for very short text

        try:
            lang = detect(text)
            # Map to our supported languages
            if lang == 'hi':
                return 'hi'
            else:
                return 'en'  # Treat everything non-Hindi as English
        except Exception:
            return 'en'  # Default fallback

    def translate_to_hindi(self, text):
        """
        Translate English text to Hindi.
        
        Args:
            text: English input string
            
        Returns:
            Hindi translated string
        """
        if not text or len(text.strip()) == 0:
            return text

        try:
            result = self.google_translator.translate(text, src='en', dest='hi')
            return result.text if result else text
        except Exception as e:
            print(f"  ⚠ EN→HI translation failed: {e}")
            return text  # Return original as fallback

    def translate_to_english(self, text):
        """
        Translate Hindi text to English.
        
        Args:
            text: Hindi input string
            
        Returns:
            English translated string
        """
        if not text or len(text.strip()) == 0:
            return text

        try:
            result = self.google_translator.translate(text, src='hi', dest='en')
            return result.text if result else text
        except Exception as e:
            print(f"  ⚠ HI→EN translation failed: {e}")
            return text  # Return original as fallback

    def process_input(self, text):
        """
        Detect language and translate to Hindi if needed.
        
        Returns:
            (hindi_text, original_language)
        """
        lang = self.detect_language(text)

        if lang == 'hi':
            return text, 'hi'
        else:
            hindi_text = self.translate_to_hindi(text)
            return hindi_text, 'en'

    def process_output(self, hindi_response, target_language):
        """
        Translate Hindi response to target language if needed.
        
        Args:
            hindi_response: Response in Hindi
            target_language: 'hi' or 'en'
            
        Returns:
            Response in target language
        """
        if target_language == 'hi':
            return hindi_response
        else:
            return self.translate_to_english(hindi_response)


if __name__ == "__main__":
    # Quick test
    translator = Translator()

    test_texts = [
        "What are the benefits of Ashwagandha?",
        "अश्वगंधा के फायदे क्या हैं?",
        "How to balance Vata dosha?",
        "त्रिफला क्या है?",
    ]

    for text in test_texts:
        lang = translator.detect_language(text)
        print(f"\n📝 Input: {text}")
        print(f"   Language: {lang}")
        
        hindi_text, orig_lang = translator.process_input(text)
        print(f"   Hindi: {hindi_text}")
        
        if orig_lang == 'en':
            back_to_en = translator.translate_to_english(hindi_text)
            print(f"   Back to EN: {back_to_en}")
