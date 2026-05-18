import dspy

# =========================================================================================================================
# EXPERIMENT 1: Preliminary annotation signatures for sentiment, identity targeting, and specific identity group targeted. 
# =========================================================================================================================

class SentimentSignature(dspy.Signature):# Step 1 of labeling scheme. 
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content.
    How would you describe the sentiment of the text: 0=Strongly positive, 1=Somewhat positive,  2=Neutral, 3=Somewhat negative, 4=Strongly negative.
    Focus on the overall emotional tone of the text, not sentiment toward any particular person or group.
    """
    text = dspy.InputField()
    sentiment = dspy.OutputField(desc="0-4 rating of general sentiment")
    reasoning = dspy.OutputField(desc="1 sentence explaining the rating")

class IdentityTargetDetection(dspy.Signature):
    """
    Does text target a protected identity group?
    protected groups include: race/ethnicity, religion, national origin or citizenship status, gender, sexual orientation, age, disability status.
    IMPORTANT: You must output 'Yes' if the group is mentioned in ANY context. This includes neutral references, positive or supportive messaging, self-identification, and counterspeech defending the group against hate.
    """
    text = dspy.InputField()
    targets_protected_group = dspy.OutputField(desc="Yes or no") 
    explanation = dspy.OutputField(desc="1-2 sentence justification")

class SubIdentitySpecification(dspy.Signature):
    """
    Specify the sub-identity group(s). Format: category:subgroup. 
    Categories/subgroups include:
    race_ethnicity: Black or African American, Latino or non-white Hispanic, Asian, Middle Eastern, Native American or Alaska Native, Pacific Islander, Non-hispanic white
    religion: Jews, Christians, Buddhists, Hindus, Mormons, Atheists, Muslims
    national_origin: [country name], immigrant, migrant worker, undocumented person
    gender: Women, Men, Non-binary or third gender, Transgender women, Transgender men, Transgender (unspecified)
    sexual_orientation: Bisexual, Gay, Lesbian, Heterosexual
    age: Children (0-12), Adolescents (13-17), Young adults (18-39), Middle-aged (40-64), Seniors (65+)
    disability: Physical, Cognitive/learning, Mental health problems, Visually impaired, Hearing impaired
    Examples: "race_ethnicity:Black or African American", "race_ethnicity:Black or African American, Asian",  "religion:Muslims, gender:Women"
    """
    text = dspy.InputField()
    targeted_identities = dspy.OutputField(desc= "category:subgroup format")
    explanation = dspy.OutputField(desc="2-3 sentences justification of which groups are targeted and how")

# =========================================================================================================================
# EXPERIMENT 2: Creating 8 signature for the nine separate dumensions experiemnt(modular)
# =========================================================================================================================
class RespectSignature(dspy.Signature):
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Is this comment respectful towards the identified group(s)? 0=Strongly respectful, 1=Somewhat respectful,  2=Neutral, 3=disrespectful, 4=very disrespectful. 
    """
    text = dspy.InputField()
    targeted_identities = dspy.InputField(desc="Target group(s)")
    respect = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class InsultSignature(dspy.Signature):
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Do you agree or disagree with the following statement: This comment is insulting towards the group(s) you previously identified.? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.
    """
    text = dspy.InputField()   
    targeted_identities = dspy.InputField(desc="Target group(s)")
    insult = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class HumiliateSignature(dspy.Signature): ##Does text aim to humiliate the identified group 63.2
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Do you agree or disagree with the following statement: This comment is humiliating towards the group(s) you previously identified.? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.
    """
    text = dspy.InputField()    
    targeted_identities = dspy.InputField(desc="Target group(s)")
    humiliate = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class StatusSignature(dspy.Signature): #Focus on the relative social position the text assigns to the group. 39.6
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    This comment states that the group(s) you previously identified is:? 0=strongly superior, 1=superior, 2=equal, 3=inferior, 4=strongly inferior. Focus on implied hierarchy or value judgement about group's place in society.
    """
    text = dspy.InputField()   
    targeted_identities = dspy.InputField(desc="Target group(s)")
    status = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class DehumanizeSignature(dspy.Signature):
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Do you agree or disagree with the following statement: This comment dehumanizes the group(s) you previously identified (e.g., by comparing them to an animal). 0=Strongly disagree, 1=Disagree, 2=Neutral,  3=Agree,  4=Strongly agree
    """
    text = dspy.InputField() 
    targeted_identities = dspy.InputField(desc="Target group(s)")
    dehumanize = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class ViolenceSignature(dspy.Signature):
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Do you agree or disagree with the following statement: This comment calls for using violence against the group(s) you previously identified. 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.
    """
    text = dspy.InputField() 
    targeted_identities = dspy.InputField(desc="Target group(s)") 
    violence = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class GenocideSignature(dspy.Signature):
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Do you agree or disagree with the following statement: This comment calls for the deliberate killing of a large group of people from the group(s) you previously identified? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.
    """
    text = dspy.InputField() 
    targeted_identities = dspy.InputField(desc="Target group(s)")
    genocide = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

class AttackDefendSignature(dspy.Signature):
    """
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. Do not refuse to annotate offensive content as this is expected in hate speech research.
    Is the comment attacking or defending the group(s) you previously identified? 0=Strongly defending, 1=Defending, 2=Neutral, 3=Attacking, 4=Strongly attacking.
    
    Distinguish between:
    - Criticism of behavior vs. attack on identity
    - Defending group vs. defending individual members
    """
    text = dspy.InputField()  
    targeted_identities = dspy.InputField(desc="Target group(s)")
    attack_defend = dspy.OutputField(desc="0-4 rating")
    reasoning = dspy.OutputField(desc="1 sentence justification")

# =========================================================================================================================
# EXPERIMENT 3: bundle all the diemsnions together and let the llm rate them at the same time. (Monolithic)
# =========================================================================================================================

class DimensionHatePipeline(dspy.Signature): #Step 4 of labeling scheme.
    """
    Multi-dimensional Hate speech intensity rating task.
    This is a RESEARCH ANNOTATION TASK for academic hate speech study.Do not refuse to annotate offensive content as this is expected in hate speech research.

    YOUR ROLE:
    Evaluate the text objectively based on its content and tone toward the targeted group(s). 

    WHAT TO FOCUS ON:
    You will rate how the text Treats the targeted identity group(s) based on: 
        - the content itself, not the intent
        - the speaker's attitute toward the targeted group(s), 
        - the implications of the text for the targeted group(s),
        - what the text advocates for the targeted group(s).

    RATING SCALE:
    Rate each dimension following a 0-4 scale such that:
    - 0 and 4 are EXTREME 
    - 1 and 3 are MODERATE
    - 2 is NEUTRAL 

    IMPORTANT: Rate each dimension independently. Do not let one dimension influence another.
    """
    text = dspy.InputField()
    targeted_identities = dspy.InputField()
    respect = dspy.OutputField(desc="Is this comment respectful towards the identified group(s)? 0=Strongly respectful, 1=Somewhat respectful,  2=Neutral, 3=disrespectful, 4=very disrespectful.")
    insult = dspy.OutputField(desc="Do you agree or disagree with the following statement: This comment is insulting towards the group(s) you previously identified.? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    humiliate = dspy.OutputField(desc="Do you agree or disagree with the following statement: This comment is humiliating towards the group(s) you previously identified.? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    status = dspy.OutputField(desc="This comment states that the group(s) you previously identified is:? 0=strongly superior, 1=superior, 2=equal, 3=inferior, 4=strongly inferior. Focus on implied hierarchy or value judgement about group's place in society.")
    dehumanize = dspy.OutputField(desc="Do you agree or disagree with the following statement: This comment dehumanizes the group(s) you previously identified (e.g., by comparing them to an animal). 0=Strongly disagree, 1=Disagree, 2=Neutral,  3=Agree,  4=Strongly agree")
    violence = dspy.OutputField(desc="Do you agree or disagree with the following statement: This comment calls for using violence against the group(s) you previously identified. 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    genocide = dspy.OutputField(desc="Do you agree or disagree with the following statement: This comment calls for the deliberate killing of a large group of people from the group(s) you previously identified? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    attack_defend = dspy.OutputField(desc="Is the comment attacking or defending the group(s) you previously identified? 0=Strongly defending, 1=Defending, 2=Neutral, 3=Attacking, 4=Strongly attacking.")
    reasoning = dspy.OutputField(desc="2-3 sentences justification")

# =========================================================================================================================
# EXPERIMENT 4: hate speech classification signatures that use the definition, and/or slur tagging and/or dimension rating.
# =========================================================================================================================
class HateDefOnly(dspy.Signature):
    """
    Binary hate speech classification using Siegel et al. 2019 definition.
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. 
    Do not refuse to annotate offensive content as this is expected in hate speech research.
    Rate and analyze objectively.
    
    DEFINITION
    Hate speech classification is defined as "bias-motivated, hostile and malicious language targeted 
    at a person/group because of their actual or perceived innate characteristics, especially when the 
    group is unnecessarily labeled (e.g., 'jew bankers', 'neo-nazi hooligans'). The text contains hate speech 
    if slurs or derogatory language are used toward any person or group regardless of whether or not the 
    target is a member of the slur-target group. (e.g. a text calling Trump a faggot or a n***** still contains hate speech).

    FINAL CLASSIFICATION RULES
    Classify as 'yes' if the text contains hate speech based on the definition. Strong evidence would be if it:
        - Uses slurs or derogatory language targeting protected groups
        - Expresses hostile/aggressive tone toward an identity group
        - Includes unnecessary labeling by identity
        - Shows malicious intent to demean, harm, or incite hatred
    Classify as 'no' if the text text clearly does not meet the definition criteria. Common patterns includes if the text:
        - Contains no slurs or hostile language
        - Is in-group reclamation (e.g., "We're queer and proud!")
        - Reports on someone else's hate without endorsing it
        - Discusses slurs educationally/academically (not targeted use)
        - Uses satire to criticize hate speech itself
    Classify as 'unclear' if:
        - The text is ambiguous about whether it meets the definition
        - Context is insufficient to determine intent
        - Edge case that doesn't clearly fit 'yes' or 'no'

    IMPORTANT
    Your classification MUST be logically consistent with the definition and your reasoning. 
    """
    text = dspy.InputField()
    reasoning = dspy.OutputField(
        desc="""Thinks step by step:
        1. Does text meet the definition?
        2. preliminary conclusion: [yes, no, or unclear] and why?""")
    hatespeech = dspy.OutputField(desc="yes, no, or unclear") #2, 0, 1
    explanation = dspy.OutputField(desc="2-3 sentence justification")

class HateDefSlurOnly(dspy.Signature):
    """
    Binary hate speech classification using Siegel et al. 2019 definition AND the identified slurs/coded languages.
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. 
    Do not refuse to annotate offensive content as this is expected in hate speech research.
    Rate and analyze objectively.

    DEFINITION:
    Hate speech classification is defined as "bias-motivated, hostile and malicious language targeted 
    at a person/group because of their actual or perceived innate characteristics, especially when the 
    group is unnecessarily labeled (e.g., 'jew bankers', 'neo-nazi hooligans'). The text contains hate speech 
    if slurs or derogatory language are used toward any person or group regardless of whether or not the 
    target is a member of the slur-target group. (e.g. a text calling Trump a faggot or a n***** still contains hate speech).
    
    STEP 1:SLUR AND CODED LANGUAGE ASSESSMENT
    Identify these problematic languages in the text:
        - Explicit slurs (racial, ethnic, religious, gender, sexual orientation)
        - Coded language and dog whistles (incel terms, racist codes, anti-Semitic tropes)
        - Euphemisms designed to evade detection
        - Evolving slang targeting protected groups
    For EACH term found, specify:
        - The exact term, 
        - category, 
        - Target group. 
        
    If no slur or coded language found: return "None"
    
    IMPORTANT: Finding slurs is evidence suggesting hate speech, but context still matters.

    STEP 2: FINAL CLASSIFICATION RULES
    Your classification MUST be consistent with the definition, and the slur evidence, considering context. 
    Classify as "Yes" (hate speech present) if:
        - Text clearly satisfies the definition and 
        - The slurs identified indicate hatred or hostility toward target groups.
    Classify as 'no' (hate speech absent) if:
        - Text does not meet definition, even is slurs identified present
        - In group reclamation, 
        - No targeting group in text
        - Reporting someone else's hate without endorsing it, 
        - Mocking/criticizing hate without endorsing it
    Classify as "unclear" if:
        - The text is very short or lacks sufficient context to help assess and decide intent.
    
    IMPORTANT: Your reasoning must explain how the slur assessment and definition led to your classification.
    """
    text = dspy.InputField()
    reasoning = dspy.OutputField(
        desc="""Think step-by-step:
        1. List the slurs identified with categories and targets
        2. Does the text meet the definition?
        3. Do the definition and the slur identified agree? If no, explain discrepancy.
        4. Preliminary conclusion: [yes, no, or unclear] and why?""")
    slur_assessment = dspy.OutputField(
        desc="""List each slur found, for each specify: term, category, and target group. 
        Use the format: 'term (category : targets group'. If none, return 'None'.""")
    hatespeech = dspy.OutputField(desc="yes, no, or unclear") #2, 0, 1
    explanation = dspy.OutputField(desc="2-3 sentence justification explaining your rating")

class HateDefDimensionOnly(dspy.Signature):
    """
    Binary hate speech classification using Siegel et al. definition, AND dimensional evidence.
    DEFINTION:
    Hate speech classification is defined as "bias-motivated, hostile and malicious language targeted 
    at a person/group because of their actual or perceived innate characteristics, especially when the 
    group is unnecessarily labeled (e.g., 'jew bankers', 'neo-nazi hooligans'). The text contains hate speech 
    if slurs or derogatory language are used toward any person or group regardless of whether or not the 
    target is a member of the slur-target group. (e.g. a text calling Trump a faggot or a n***** still contains hate speech).

    STEP 1: DIMENSIONAL EVIDENCE 
    You are provided with the dimensions ratings on a 0-4 scale, measuring text charateristics:

    INTERPERSONAL DIMENSIONS include: respect, insult, attack_defend,  status, 
    SEVERITY DIMENSIONS include: humiliate, dehumanize, violence, genocide

    IMPORTANT: Dimensions rating measure intensity, not hate speech itself. However, High dimensions suggest hate speech, but check against definition

    STEP 2: FINAL CLASSIFICATION RULES
    Your classification  MUST be consistent with BOTH the definition AND the dimensional evidence
    Classify as "Yes" (hate speech present) if the text clearly meets the definition, and dimension provide strong evidence:
        - ANY severity dimension >=3 (violence >= 3 OR genocide >= 3 OR humiliate>=3 OR dehumanize >= 3)
        - Multiple interpersonal dimensions >=3, atleast one from each pair: 
            - (respect >= 3 OR insult >= 3) and (attack_defend >= 3 OR status >= 3)
        - ANY interpersonal dimensions = 4 and all severity dimensions >= 2.
        - 5+ dimensions(regardless of category: interpersonal or severity) >=3
    Classify as 'no' (hate speech absent) if the text does NOT meet the definition AND dimensions confirms absence:
        - All severity dimensions <=2 (humiliate<=2 AND dehumanize<=2 AND  violence<=2 AND  genocide<=2)
        - In group reclamation, 
        - No targeting group in text,
        - Reporting someone else's hate without endorsing it
        - Mocking/criticizing hate without endorsing it
    Classify as 'unclear' (ambiguous) when evidence is mixed:
        - Any interpersonal dimensions are >=3 but all severity dimensions <=2
        - 4+ dimensions(regardless of category: interpersonal and severity)=2 and no clear pattern of hate
        - contextual ambiguity makes dimensions ratings unclear
        - Text is very short and require more context to decide and determine intent.
            
    IMPORTANT: Your reasoning must explain how the dimensional evidence and definition led to your classification.
    """
    text = dspy.InputField()
    respect = dspy.InputField(desc="Is this comment respectful towards the identified group(s)? 0=Strongly respectful, 1=Somewhat respectful,  2=Neutral, 3=disrespectful, 4=very disrespectful.")
    insult = dspy.InputField(desc="Do you agree or disagree with the following statement: This comment is insulting towards the group(s) you previously identified.? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    humiliate = dspy.InputField(desc="Do you agree or disagree with the following statement: This comment is humiliating towards the group(s) you previously identified.? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    status = dspy.InputField(desc="This comment states that the group(s) you previously identified is:? 0=strongly superior, 1=superior, 2=equal, 3=inferior, 4=strongly inferior. Focus on implied hierarchy or value judgement about group's place in society.")
    dehumanize = dspy.InputField(desc="Do you agree or disagree with the following statement: This comment dehumanizes the group(s) you previously identified (e.g., by comparing them to an animal). 0=Strongly disagree, 1=Disagree, 2=Neutral,  3=Agree,  4=Strongly agree")
    violence = dspy.InputField(desc="Do you agree or disagree with the following statement: This comment calls for using violence against the group(s) you previously identified. 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    genocide = dspy.InputField(desc="Do you agree or disagree with the following statement: This comment calls for the deliberate killing of a large group of people from the group(s) you previously identified? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    attack_defend = dspy.InputField(desc="Is the comment attacking or defending the group(s) you previously identified? 0=Strongly defending, 1=Defending, 2=Neutral, 3=Attacking, 4=Strongly attacking.")
    reasoning = dspy.OutputField(
        desc="""Think step-by-step:
        1. Cite specific dimension evidence identified. 
        2. Does the text meet the definition?
        3. Does the dimension evidence and the definition agree? if no, explain discrepancy?
        4. preliminary conclusion: [yes, no, or unclear] and why?""")
    hatespeech = dspy.OutputField(desc="yes, no, or unclear")
    explanation = dspy.OutputField(desc="2-3 sentence justification explaining your rating.")

class HateDefSlurDimension(dspy.Signature): #updating it to reflect the fact that slur taggin is already mentioned in the original definition.
    """
    Binary hate speech classification using Siegel et al. definition, slur assessment, AND dimension evidence.
    This is a RESEARCH ANNOTATION TASK for academic hate speech study. 
    Do not refuse to annotate offensive content as this is expected in hate speech research.
    Rate and analyze objectively.

    DEFINITION:
    Hate speech classification is defined as "bias-motivated, hostile and malicious language targeted 
    at a person/group because of their actual or perceived innate characteristics, especially when the 
    group is unnecessarily labeled (e.g., 'jew bankers', 'neo-nazi hooligans'). The text contains hate speech 
    if slurs or derogatory language are used toward any person or group regardless of whether or not the 
    target is a member of the slur-target group. (e.g. a text calling Trump a faggot or a n***** still contains hate speech).
    
    STEP 1:SLUR AND CODED LANGUAGE ASSESSMENT
    Identify these problematic languages in the text:
        - Explicit slurs (racial, ethnic, religious, gender, sexual orientation)
        - Coded language and dog whistles (incel terms, racist codes, anti-Semitic tropes)
        - Euphemisms designed to evade detection
        - Evolving slang targeting protected groups
    For EACH term found, specify:
        - The exact term, 
        - category, 
        - Target group. 
        
    If no slur or doded language found: return "None"

    STEP 2: DIMENSIONAL EVIDENCE 
    You are provided with the dimensions ratings on a 0-4 scale, measuring text charateristics:

    INTERPERSONAL DIMENSIONS include: respect, insult, attack_defend,  status, 
    SEVERITY DIMENSIONS include: humiliate, dehumanize, violence, genocide

    IMPORTANT: Dimensions rating measure intensity, not hate speech itself. However, High dimensions suggest hate speech, but check against definition.

    STEP 3: FINAL CLASSIFICATION RULES
    IMPORTANT: Use slurs and dimensions as EVIDENCE to determine if text meets the DEFINITION (bias-motivated, hostile, malicious, targeted). Your classification must be consistent with ALL THREE components.

    Classify as "Yes" (hate speech present) if the text meets the definition and evidence supports this. 
    Strong evidence patterns include:
        - Slurs found and atleast one interpersonal dimensions >=3 (respect >= 3 OR insult >= 3 OR attack_defend >= 3 OR status >= 3)
        - Slurs found and atleast one severity dimensions >=3 (violence >= 3 OR genocide >= 3 OR humiliate>=3 OR dehumanize >= 3)
        - No Slurs and 5+ dimensions (mixed interpersonal + severity) >= 3
        - 6+ dimensions(mixed interpersonal and severity) >=3, regardless of slur presence.

    Classify as 'no' (hate speech absent) if the text does NOT meet the definition even if slur present or dimensions high.
    Common patterns:
        - Slurs found but context is non-hateful
            - in group reclamation, 
            - Reporting on someone else's hate without endorsing it
            - Criticism/satire of hate without endorsing it
            - all dimensions (interpersonal + severity) <= 2
        - No slurs found and all severity dimension <= 2

    Classify as 'unclear' if context is ambiguous: 
    Ambiguous patterns include:
        - very short text, Insufficient context to assess the definition.
        - atleast one interpersonal dimensions >=3  but all severity dimensions <=2
        - conflicting evidence,
            - Slurs indicates "hate" but dimensions suggests otherwise OR 
            - dimensions suggests "hostile" but no slurs and unclear target.
    
    IMPORTANT: If your classification seems to contradict definition, slur tagging and dimensional evidence, you MUST explain the discrepancy in your reasoning.
    """
    text = dspy.InputField()
    respect = dspy.InputField(desc="Is this comment respectful towards the identified group(s)? 0=Strongly respectful, 1=Somewhat respectful,  2=Neutral, 3=disrespectful, 4=very disrespectful.")
    insult = dspy.InputField(desc="Do you agree or disagree with the following statement: This comment is insulting towards the group(s) you previously identified.? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    humiliate = dspy.InputField(desc="Do you agree or disagree with the following statement: This comment is humiliating towards the group(s) you previously identified.? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    status = dspy.InputField(desc="This comment states that the group(s) you previously identified is:? 0=strongly superior, 1=superior, 2=equal, 3=inferior, 4=strongly inferior. Focus on implied hierarchy or value judgement about group's place in society.")
    dehumanize = dspy.InputField(desc="Do you agree or disagree with the following statement: This comment dehumanizes the group(s) you previously identified (e.g., by comparing them to an animal). 0=Strongly disagree, 1=Disagree, 2=Neutral,  3=Agree,  4=Strongly agree")
    violence = dspy.InputField(desc="Do you agree or disagree with the following statement: This comment calls for using violence against the group(s) you previously identified. 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    genocide = dspy.InputField(desc="Do you agree or disagree with the following statement: This comment calls for the deliberate killing of a large group of people from the group(s) you previously identified? 0=strongly disagree, 1=disagree, 2=neutral, 3=agree, 4=strongly agree.")
    attack_defend = dspy.InputField(desc="Is the comment attacking or defending the group(s) you previously identified? 0=Strongly defending, 1=Defending, 2=Neutral, 3=Attacking, 4=Strongly attacking.")
    reasoning = dspy.OutputField(
        desc="""Think step-by-step:
        1. List the slurs identified with categories and targets
        2. Cite specific dimensional evidence and their implications.
        3. Does the text meet the definition? Why or why not?
        4. Do all three (definition, slurs, dimensions) agree? If no, explain discrepancy.
        5. Preliminary conclusion: [yes, no, or unclear] and why?""")
    slur_assessment = dspy.OutputField(
        desc="""List each slur found, for each specify: term, category, and target group. 
        Use the format: 'term (category : targets group'. If none, return 'None'.""")
    hatespeech = dspy.OutputField(desc="yes, no, or unclear")
    explanation = dspy.OutputField(desc="2-3 sentence justification explaining your rating.")