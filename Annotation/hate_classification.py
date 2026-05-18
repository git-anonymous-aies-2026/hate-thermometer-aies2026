import dspy

from signatures import (
    DimensionHatePipeline,
    HateDefOnly, 
    HateDefSlurOnly, 
    HateDefDimensionOnly, 
    HateDefSlurDimension

)
# ======================================================================================================================
# EXPERIMENT 1: Comparing Definition-Only vs Definition-SlurTagging vs Definition+slurtagging+Dimensions implementations
# =======================================================================================================================

class HateClassificationComparison(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rate_dimensions = dspy.ChainOfThought(DimensionHatePipeline)
        self.rate_hate_def_only = dspy.ChainOfThought(HateDefOnly) #step 5a: Hate speech classification using only definition (for non protected group targeting cases)
        self.rate_hate_def_slur = dspy.ChainOfThought(HateDefSlurOnly) #step 5b: Hate speech classification using definition + slur tagging, only if it performs better than definition only)
        self.rate_hate_def_dim = dspy.ChainOfThought(HateDefDimensionOnly) # Step 5c: Binary hate speech or HateClassificationWithDimensions (for protected group targeting cases)
        self.rate_hate_def_slur_dim = dspy.ChainOfThought(HateDefSlurDimension) # Step 5d: Binary hate speech or HateClassificationWithDimensions (for protected group targeting cases)

    def forward(self, text, slur_tagged_text=None):
        text_for_analysis = slur_tagged_text if slur_tagged_text else text

        dimensions = self.rate_dimensions(
            text=text_for_analysis,
            targeted_identities='yes' #since we are providing the dimensions, we can just say yes to targeted identities, and let the dimension rater determine the specific ratings
        )
        classification_def_only = self.rate_hate_def_only(text=text_for_analysis)
        classification_def_slur = self.rate_hate_def_slur(text=text_for_analysis)
        classification_def_dim = self.rate_hate_def_dim(
            text=text_for_analysis,
            respect=dimensions.respect,                             
            insult=dimensions.insult,
            humiliate=dimensions.humiliate,
            status=dimensions.status,
            dehumanize=dimensions.dehumanize,
            violence=dimensions.violence,
            genocide=dimensions.genocide,
            attack_defend=dimensions.attack_defend)
        classification_def_slur_dim = self.rate_hate_def_slur_dim(
            text=text_for_analysis,
            respect=dimensions.respect,                             
            insult=dimensions.insult,
            humiliate=dimensions.humiliate,
            status=dimensions.status,
            dehumanize=dimensions.dehumanize,
            violence=dimensions.violence,
            genocide=dimensions.genocide,
            attack_defend=dimensions.attack_defend )

        return dspy.Prediction(
        # Method 1: Definition only
        hatespeech_def_only=classification_def_only.hatespeech,
        explanation_def_only=classification_def_only.explanation,
        reasoning_def_only=classification_def_only.reasoning,
        
        # Method 2: Definition + Slur
        hatespeech_def_slur=classification_def_slur.hatespeech,
        explanation_def_slur=classification_def_slur.explanation,
        reasoning_def_slur=classification_def_slur.reasoning,
        slur_assessment_def_slur=classification_def_slur.slur_assessment,  
        
        # Method 3: Definition + Dimensions
        hatespeech_def_dim=classification_def_dim.hatespeech,
        explanation_def_dim=classification_def_dim.explanation,
        reasoning_def_dim=classification_def_dim.reasoning,
        
        # Method 4: Definition + Slur + Dimensions (Full)
        hatespeech_def_slur_dim=classification_def_slur_dim.hatespeech,
        explanation_def_slur_dim=classification_def_slur_dim.explanation,
        reasoning_def_slur_dim=classification_def_slur_dim.reasoning,
        slur_assessment_def_slur_dim=classification_def_slur_dim.slur_assessment,
        
        # Dimensional ratings
        # llm_sentiment=dimensions.sentiment,
        llm_respect=dimensions.respect,
        llm_insult=dimensions.insult,
        llm_humiliate=dimensions.humiliate,
        llm_status=dimensions.status,
        llm_dehumanize=dimensions.dehumanize,
        llm_violence=dimensions.violence,
        llm_genocide=dimensions.genocide,
        llm_attack_defend=dimensions.attack_defend)
