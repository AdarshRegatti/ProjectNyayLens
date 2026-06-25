"""
Enhanced Judgment Segmenter (FIXED)
Segments judgments into: Facts, Issues, Arguments, Analysis, Decision
"""

import re
import os
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Section:
    type: str                  # facts/issues/arguments/analysis/decision/unknown
    text: str
    start_para_idx: int
    end_para_idx: int
    confidence: float


class JudgmentSegmenter:

    MARKERS = {
        'facts': [
            r'\bbrief\s+facts?\b',
            r'\bfactual\s+(matrix|background)\b',
            r'\bcircumstances\s+of\s+the\s+case\b',
            r'\bbackground\b',
        ],
        'issues': [
            r'\bissues?\s+(for|of)\s+(consideration|determination)\b',
            r'\bsubstantial\s+questions?\b',
            r'\bpoints?\s+for\s+consideration\b',
            r'\bquestions?\s+framed\b',
        ],
        'arguments': [
            r'\blearned\s+counsel\b',
            r'\bsubmissions?\b',
            r'\b(argued|submitted|contended)\b',
            r'\bon\s+behalf\s+of\b',
        ],
        'analysis': [
            r'\bwe\s+have\s+(considered|examined|analysed)\b',
            r'\bthe\s+court\s+(finds|observes|notes|holds)\b',
            r'\bin\s+our\s+(view|opinion)\b',
            r'\bit\s+is\s+clear\s+that\b',
        ],
        'decision': [
            r'\b(appeal|petition|writ)\s+is\s+(allowed|dismissed)\b',
            r'\baccordingly\b',
            r'\bwe\s+direct\b',
            r'\bheld\s*:\b',
            r'\border\b',
        ]
    }

    def __init__(self, model_path: str = "models/segmentation_model"):
        """Initialize segmenter, preferring ML model if available, else Regex fallback"""
        self.use_ml = False
        self.classifier = None
        
        if TRANSFORMERS_AVAILABLE and os.path.exists(model_path):
            try:
                logger.info(f"Loading ML Segmentation model from {model_path}...")
                self.classifier = pipeline("text-classification", model=model_path, device=-1)
                self.use_ml = True
                logger.info("✓ ML Segmenter loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load ML model, falling back to Regex: {e}")
        else:
            logger.info("ML model not found or transformers not installed. Using Regex fallback.")

    def detect_section(self, para: str, position_ratio: float) -> Tuple[str, float]:
        """
        Detect section type for a paragraph
        Returns: (section_type, confidence)
        """
        para_lower = para.lower()
        best_type = 'unknown'
        best_conf = 0.0

        for sec_type, patterns in self.MARKERS.items():
            for pattern in patterns:
                if re.search(pattern, para_lower):
                    conf = 0.6

                    # Position-based bias
                    if sec_type == 'facts' and position_ratio < 0.30:
                        conf += 0.2
                    elif sec_type == 'decision' and position_ratio > 0.70:
                        conf += 0.3

                    # Strong anchor near paragraph start
                    if re.search(pattern, para_lower[:120]):
                        conf += 0.2

                    conf = min(conf, 1.0)

                    if conf > best_conf:
                        best_type = sec_type
                        best_conf = conf

        return best_type, best_conf
        
    def detect_section_ml(self, para: str) -> Tuple[str, float]:
        """Detect using HuggingFace classifier"""
        if not para.strip() or not self.classifier:
            return "unknown", 0.0
            
        # Truncate to max length to avoid tokenization errors
        truncated = para[:512]
        result = self.classifier(truncated)[0]
        
        # Assume labels are like LABEL_FACTS, LABEL_ISSUES or directly facts, issues
        label = result['label'].lower().replace('label_', '')
        score = result['score']
        
        # Enforce confidence threshold
        if score < 0.5:
            return "unknown", score
            
        return label, score

    def segment(self, paragraph_texts: List[str]) -> List[Section]:
        """
        Segment judgment based on paragraph list (INDEX-ALIGNED)
        """
        if not paragraph_texts:
            return []

        sections: List[Section] = []

        current_type = 'unknown'
        current_paras = []
        current_conf = 0.0
        start_idx = 0

        total = len(paragraph_texts)

        for i, para in enumerate(paragraph_texts):
            position_ratio = i / max(total, 1)
            
            if self.use_ml:
                sec_type, conf = self.detect_section_ml(para)
            else:
                sec_type, conf = self.detect_section(para, position_ratio)

            # Fallback: early unknown paragraphs are likely facts
            if sec_type == 'unknown' and position_ratio < 0.30 and i > 0:
                sec_type = 'facts'
                conf = 0.4

            # Section boundary
            if conf > 0.4 and sec_type != current_type:
                if current_paras:
                    sections.append(
                        Section(
                            type=current_type,
                            text="\n\n".join(current_paras),
                            start_para_idx=start_idx,
                            end_para_idx=i - 1,
                            confidence=round(current_conf, 2)
                        )
                    )

                current_type = sec_type
                current_paras = [para]
                current_conf = conf
                start_idx = i
            else:
                current_paras.append(para)
                current_conf = max(current_conf, conf)

        # Final section
        if current_paras:
            sections.append(
                Section(
                    type=current_type,
                    text="\n\n".join(current_paras),
                    start_para_idx=start_idx,
                    end_para_idx=total - 1,
                    confidence=round(current_conf, 2)
                )
            )

        return sections
