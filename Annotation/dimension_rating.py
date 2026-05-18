import dspy, os
from dspy.predict import Predict
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

#importing signatured ffor the hate classification
from signatures import (
    SentimentSignature,
    RespectSignature,
    InsultSignature,
    HumiliateSignature,
    StatusSignature,
    DehumanizeSignature,
    ViolenceSignature,
    GenocideSignature,
    AttackDefendSignature,
    IdentityTargetDetection,
    SubIdentitySpecification,
    HateDefSlurOnly
)

#importing signatures for the dimension ratings in the monolithic model
from signatures import DimensionHatePipeline   

# ===========================================================================================
# EXPERIMENT 1: MONOLITHIC - Comparing 9 dimensions in one signature call
# ============================================================================================

class RatingNineDimensionsTogether(dspy.Module):
    """
    All 9 dimensions in ONE signature call.
    """
    def __init__(self):
        super().__init__()
        self.rate_sentiment = dspy.ChainOfThought(SentimentSignature) # Step 1: Sentiment baseline
        self.identify_targets = dspy.ChainOfThought(IdentityTargetDetection) # Step 2: Identity target detection
        self.specify_identities = dspy.ChainOfThought(SubIdentitySpecification) #Step 3: Sub-identity specification (conditional on step 2)
        self.rate_dimensions = dspy.ChainOfThought(DimensionHatePipeline) # Step 4: Remaining scale items only if step 2 not none
        self.rate_hate_def_slur = dspy.ChainOfThought(HateDefSlurOnly) #step 5b: Hate speech classification using definition + slur tagging, only if it performs better than definition only)

    def forward(self, text, slur_tagged_text=None):
        text_for_analysis = slur_tagged_text if slur_tagged_text else text
        sentiment_result = self.rate_sentiment(text=text_for_analysis)
        targeting_result = self.identify_targets(text=text_for_analysis)

        if targeting_result.targets_protected_group.lower() == 'yes':
            # Step 3: sub-categories
            identity_result = self.specify_identities(text=text_for_analysis)
            dimensions_result = self.rate_dimensions(text=text_for_analysis,targeted_identities=identity_result.targeted_identities )
            
            # classification_def_only = self.rate_hate_def_only(text=text_for_analysis)
            classification_def_slur = self.rate_hate_def_slur(text=text_for_analysis)

            return dspy.Prediction(
                original_text=text,  
                slur_tagged_text=slur_tagged_text,
                sentiment=sentiment_result.sentiment,
                targets_protected_group='yes',
                targeting_explanation=targeting_result.explanation,
                targeted_identities=identity_result.targeted_identities,
                identity_explanation=identity_result.explanation,
                #step 4: monolithic
                respect=dimensions_result.respect,
                insult=dimensions_result.insult,
                humiliate=dimensions_result.humiliate,
                status=dimensions_result.status,
                dehumanize=dimensions_result.dehumanize,
                violence=dimensions_result.violence,
                genocide=dimensions_result.genocide,
                attack_defend=dimensions_result.attack_defend,
                dimensional_reasoning=dimensions_result.reasoning,
                
                # Step 5b: Classification - Definition + Slur
                hatespeech=classification_def_slur.hatespeech,
                explanation=classification_def_slur.explanation,
                reasoning=classification_def_slur.reasoning,
                slur_assessment=classification_def_slur.slur_assessment)
        else:
            classification_def_slur = self.rate_hate_def_slur(text=text_for_analysis) # Step 5b: Binary HateClassification With Definition + Slur Tagging, only if it performs better than definition only
            return dspy.Prediction(
                original_text=text,  
                slur_tagged_text=slur_tagged_text,
                sentiment=sentiment_result.sentiment,
                targets_protected_group='no',
                targeting_explanation=None,
                targeted_identities=None,
                identity_explanation=None,
                respect=None,
                insult=None,
                humiliate=None,
                status=None,
                dehumanize=None,
                violence=None,
                genocide=None,
                attack_defend=None,
                dimensional_reasoning=None,
                
                # Step 5b: Classification - Definition + Slur
                hatespeech=classification_def_slur.hatespeech,
                explanation=classification_def_slur.explanation,
                reasoning=classification_def_slur.reasoning,
                slur_assessment=classification_def_slur.slur_assessment )

# ============================================================================================
# EXPERIMENT 2: MODULAR - Comparing 9 dimensions in separate signature calls
# ============================================================================================
class RatingNineDimensionsSeparate(dspy.Module):
    '''
    Rate all 9 dimensions using separate signature calls.
    '''
    def __init__(self):
        super().__init__()
        # Step 1: Sentiment baseline (same for all dimensions, regardless of targeting)
        self.rate_sentiment = dspy.ChainOfThought(SentimentSignature)
        # Step 2: Identity target detection
        self.identify_targets = dspy.ChainOfThought(IdentityTargetDetection) 
        # Step 3: Sub-identity specification (conditional on step 2)
        self.specify_identities = dspy.ChainOfThought(SubIdentitySpecification) 
        # Step 4: Separate calls for each dimension rating
        self.rate_respect = dspy.ChainOfThought(RespectSignature)
        self.rate_insult = dspy.ChainOfThought(InsultSignature)
        self.rate_humiliate = dspy.ChainOfThought(HumiliateSignature)
        self.rate_status = dspy.ChainOfThought(StatusSignature)
        self.rate_dehumanize = dspy.ChainOfThought(DehumanizeSignature)
        self.rate_violence = dspy.ChainOfThought(ViolenceSignature)
        self.rate_genocide = dspy.ChainOfThought(GenocideSignature)
        self.rate_attack_defend = dspy.ChainOfThought(AttackDefendSignature)
        # Step 5: Binary hate speech or HateClassificationWithDimensions
        self.rate_hate_def_slur = dspy.ChainOfThought(HateDefSlurOnly) #step 5b: Hate speech classification using definition + slur tagging, only if it performs better than definition only)

    def forward(self, text, slur_tagged_text=None):
        text_for_analysis  = slur_tagged_text if slur_tagged_text else text
        sentiment_result = self.rate_sentiment(text=text_for_analysis)
        targeting_result = self.identify_targets(text=text_for_analysis)
        if targeting_result.targets_protected_group.lower() == 'yes':
            identity_result = self.specify_identities(text=text_for_analysis)
            respect_result = self.rate_respect(text=text_for_analysis, targeted_identities=identity_result.targeted_identities)
            insult_result = self.rate_insult(text=text_for_analysis, targeted_identities=identity_result.targeted_identities)
            humiliate_result = self.rate_humiliate(text=text_for_analysis, targeted_identities=identity_result.targeted_identities)
            status_result = self.rate_status(text=text_for_analysis, targeted_identities=identity_result.targeted_identities)
            dehumanize_result = self.rate_dehumanize(text=text_for_analysis, targeted_identities=identity_result.targeted_identities)
            violence_result = self.rate_violence(text=text_for_analysis, targeted_identities=identity_result.targeted_identities)
            genocide_result = self.rate_genocide(text=text_for_analysis, targeted_identities=identity_result.targeted_identities)
            attack_defend_result = self.rate_attack_defend(text=text_for_analysis, targeted_identities=identity_result.targeted_identities)

            # Step 5: All 4 classification methods
            classification_def_slur = self.rate_hate_def_slur(text=text_for_analysis)

            return dspy.Prediction(
                original_text=text,  
                slur_tagged_text=slur_tagged_text,
                sentiment=sentiment_result.sentiment,
                targets_protected_group='yes',
                targeting_explanation=targeting_result.explanation,
                targeted_identities=identity_result.targeted_identities,
                identity_explanation=identity_result.explanation,
                #modular dimensions step
                respect=respect_result.respect,
                respect_reasoning=respect_result.reasoning,
                
                insult=insult_result.insult,
                insult_reasoning=insult_result.reasoning,
                
                humiliate=humiliate_result.humiliate,
                humiliate_reasoning=humiliate_result.reasoning,
                
                status=status_result.status,
                status_reasoning=status_result.reasoning,
                
                dehumanize=dehumanize_result.dehumanize,
                dehumanize_reasoning=dehumanize_result.reasoning,
                
                violence=violence_result.violence,
                violence_reasoning=violence_result.reasoning,
                
                genocide=genocide_result.genocide,
                genocide_reasoning=genocide_result.reasoning,
                
                attack_defend=attack_defend_result.attack_defend,
                attack_defend_reasoning = attack_defend_result.reasoning,

                # Step 5b: Classification - Definition + Slur
                hatespeech=classification_def_slur.hatespeech,
                explanation=classification_def_slur.explanation,
                reasoning=classification_def_slur.reasoning,
                slur_assessment=classification_def_slur.slur_assessment )
        else:
            # No targeting, Skip dimensional ratings
            classification_def_slur = self.rate_hate_def_slur(text=text_for_analysis) # Step 5b: Binary HateClassification With Definition + Slur Tagging, only if it performs better than definition only
            
            return dspy.Prediction( 
                original_text=text,  
                slur_tagged_text=slur_tagged_text,    
                sentiment=sentiment_result.sentiment,
                targets_protected_group='no',
                targeting_explanation=None,
                targeted_identities=None,
                identity_explanation=None,
                respect=None,
                respect_reasoning=None,

                insult=None,
                insult_reasoning=None,

                humiliate=None,
                humiliate_reasoning=None,

                status=None,
                status_reasoning=None,

                dehumanize=None,
                dehumanize_reasoning=None,

                violence=None,
                violence_reasoning=None,

                genocide=None,
                genocide_reasoning=None,

                attack_defend=None,
                attack_defend_reasoning=None,
                
                # Step 5b: Classification - Definition + Slur
                hatespeech=classification_def_slur.hatespeech,
                explanation=classification_def_slur.explanation,
                reasoning=classification_def_slur.reasoning,
                slur_assessment=classification_def_slur.slur_assessment)