# Epidemiology Fundamentals for Data Science

## Table of Contents

1. [Introduction to Epidemiology](#introduction-to-epidemiology)
2. [Core Epidemiological Concepts](#core-epidemiological-concepts)
3. [Study Designs in Epidemiology](#study-designs-in-epidemiology)
4. [Measures of Disease Frequency](#measures-of-disease-frequency)
5. [Measures of Association](#measures-of-association)
6. [Bias and Confounding](#bias-and-confounding)
7. [Causal Inference](#causal-inference)
8. [Outbreak Investigation](#outbreak-investigation)
9. [Surveillance Systems](#surveillance-systems)
10. [Data Science Applications](#data-science-applications)
11. [Modern Epidemiological Methods](#modern-epidemiological-methods)
12. [Computational Epidemiology](#computational-epidemiology)

## Introduction to Epidemiology

### Definition and Scope

**Epidemiology** is the study of the distribution and determinants of health-related states or events in specified populations, and the application of this study to the prevention and control of health problems.

#### Key Components:
- **Distribution**: Who, when, where (descriptive epidemiology)
- **Determinants**: Why and how (analytical epidemiology)
- **Health-related states**: Diseases, injuries, disabilities, death
- **Specified populations**: Defined groups of people
- **Application**: Prevention and control (applied epidemiology)

#### Historical Context

##### Pioneers in Epidemiology
1. **John Snow (1813-1858)**
   - Cholera outbreak investigation in London (1854)
   - Removed Broad Street pump handle
   - Father of modern epidemiology
   - Demonstrated importance of water-borne transmission

2. **Ignaz Semmelweis (1818-1865)**
   - Reduced puerperal fever through handwashing
   - Early recognition of healthcare-associated infections
   - Faced resistance to his findings

3. **Austin Bradford Hill (1897-1991)**
   - Hill's criteria for causation
   - Developed randomized controlled trials
   - Tobacco and lung cancer studies

#### Modern Evolution
- **Molecular Epidemiology**: Integration with genetics and biomarkers
- **Environmental Epidemiology**: Focus on environmental exposures
- **Social Epidemiology**: Emphasis on social determinants
- **Digital Epidemiology**: Use of digital data sources
- **Precision Public Health**: Personalized prevention strategies

### The Epidemiological Approach

#### Three Core Functions
1. **Public Health Surveillance**: Systematic collection, analysis, and interpretation of health data
2. **Field Investigation**: Systematic investigation of health problems
3. **Analytic Studies**: Testing specific hypotheses about health determinants

#### The Epidemiological Triangle
```
    Agent
     /\
    /  \
   /    \
Host----Environment
```

- **Agent**: Causative factor (pathogen, chemical, behavior)
- **Host**: Person who may develop disease
- **Environment**: External factors that affect agent and host

#### Population vs. Clinical Medicine
| Aspect | Clinical Medicine | Epidemiology |
|--------|-------------------|--------------|
| **Focus** | Individual patient | Population groups |
| **Approach** | Diagnosis and treatment | Prevention and control |
| **Perspective** | Disease-oriented | Health-oriented |
| **Method** | Case studies | Population studies |
| **Outcome** | Individual health | Population health |

### Types of Epidemiology

#### 1. Descriptive Epidemiology
**Purpose**: Describes disease patterns by person, place, and time

**Person Characteristics**:
- Age, sex, race/ethnicity
- Socioeconomic status
- Occupation
- Lifestyle factors
- Genetic factors

**Place Characteristics**:
- Geographic distribution
- Urban vs. rural
- Environmental factors
- Healthcare access

**Time Characteristics**:
- Secular trends (long-term changes)
- Seasonal patterns
- Epidemic curves
- Cyclic patterns

#### 2. Analytical Epidemiology
**Purpose**: Tests specific hypotheses about disease determinants

**Key Questions**:
- What causes disease?
- What factors increase/decrease risk?
- How effective are interventions?

**Study Types**:
- Observational studies
- Experimental studies
- Quasi-experimental studies

#### 3. Applied Epidemiology
**Purpose**: Applies epidemiological principles to solve public health problems

**Activities**:
- Outbreak investigation
- Program evaluation
- Policy development
- Surveillance system design

## Core Epidemiological Concepts

### Health and Disease

#### Disease Natural History
```
Exposure → Pathological Changes → Clinical Disease → Recovery/Disability/Death
    ↓              ↓                    ↓                    ↓
Susceptible    Subclinical         Clinical            Outcome
 Period         Period             Period              Period
```

#### Levels of Prevention

##### Primary Prevention
- **Goal**: Prevent disease occurrence
- **Target**: Healthy individuals at risk
- **Examples**: Vaccination, health education, environmental modifications
- **Timing**: Before disease develops

##### Secondary Prevention
- **Goal**: Early detection and treatment
- **Target**: Asymptomatic individuals with early disease
- **Examples**: Screening programs, regular check-ups
- **Timing**: Early disease stage

##### Tertiary Prevention
- **Goal**: Prevent complications and disability
- **Target**: Individuals with established disease
- **Examples**: Rehabilitation, disease management
- **Timing**: After disease diagnosis

#### Iceberg Concept of Disease
```
Clinical Cases (visible)
    ↑ (Tip of iceberg)
─────────────────────── Water line
Subclinical Cases
Exposed but not infected
Susceptible population
    ↓ (Below water line)
```

Most disease burden is "below the water line" and not clinically apparent.

### Population Concepts

#### Population at Risk
- **Definition**: Group of individuals who can develop the disease
- **Characteristics**: Must be susceptible to the outcome
- **Examples**: Women for cervical cancer, workers for occupational diseases

#### Population Dynamics
- **Immigration**: People entering the population
- **Emigration**: People leaving the population
- **Births**: New individuals entering
- **Deaths**: Individuals leaving through mortality

#### Age Structure Effects
- **Population Pyramid**: Graphical representation of age-sex structure
- **Demographic Transition**: Changes in birth/death rates over time
- **Standardization**: Adjusting for age differences between populations

### Causation in Epidemiology

#### Sufficient Cause Model (Rothman)
- **Sufficient Cause**: Set of minimal conditions that inevitably produce disease
- **Component Causes**: Individual factors within sufficient cause
- **Necessary Cause**: Component that appears in every sufficient cause

#### Causal Pies Diagram
```
Pie 1: A + B + C = Disease
Pie 2: A + D + E = Disease  
Pie 3: A + F + G = Disease
```
Where A is a necessary cause (appears in all pies)

#### Multi-causal Model
- **Web of Causation**: Complex interplay of multiple factors
- **Proximate Causes**: Immediate factors
- **Distal Causes**: Underlying/root causes
- **Intermediate Causes**: Factors in causal pathway

## Study Designs in Epidemiology

### Overview of Study Designs

```
Epidemiological Studies
├── Observational Studies
│   ├── Descriptive
│   │   ├── Case Reports/Series
│   │   ├── Cross-sectional
│   │   └── Ecological
│   └── Analytical
│       ├── Case-control
│       ├── Cohort (prospective/retrospective)
│       └── Nested case-control
└── Experimental Studies
    ├── Randomized Controlled Trials
    ├── Community Trials
    └── Quasi-experimental
```

### Descriptive Studies

#### Case Reports and Case Series

**Case Report**:
- **Definition**: Detailed description of a single patient
- **Purpose**: Document unusual presentations, new diseases
- **Limitations**: Cannot determine causation, no comparison group
- **Example**: First AIDS cases reported in 1981

**Case Series**:
- **Definition**: Collection of case reports with similar features
- **Purpose**: Identify patterns, generate hypotheses
- **Analysis**: Descriptive statistics only
- **Example**: Thalidomide birth defects case series

#### Cross-sectional Studies (Prevalence Studies)

**Design Features**:
- **Timing**: Exposure and outcome measured simultaneously
- **Population**: Defined population at specific time
- **Sampling**: Representative sample of population
- **Analysis**: Prevalence estimates, associations

**Advantages**:
- Quick and inexpensive
- Good for chronic diseases
- Multiple exposures and outcomes
- Useful for public health planning

**Disadvantages**:
- Cannot establish temporal relationship
- Not suitable for rare diseases
- Prevalence-incidence bias
- Survivor bias

**Example Applications**:
- National Health and Nutrition Examination Survey (NHANES)
- Behavioral Risk Factor Surveillance System (BRFSS)
- Disease prevalence surveys

#### Ecological Studies

**Design Features**:
- **Unit of Analysis**: Populations or groups
- **Data Sources**: Aggregate data
- **Exposure**: Group-level exposures
- **Outcome**: Group-level outcomes

**Types**:
1. **Exploratory**: Generate hypotheses
2. **Analytical**: Test associations
3. **Multiple-group**: Compare different populations
4. **Time-trend**: Same population over time
5. **Mixed**: Combines approaches

**Advantages**:
- Uses existing data
- Inexpensive
- Good for policy-relevant exposures
- Large effect sizes detectable

**Disadvantages**:
- Ecological fallacy
- Cannot control for individual confounders
- Temporal ambiguity
- Crude exposure assessment

**Example**: Per capita alcohol consumption vs. liver cirrhosis mortality by country

### Analytical Studies

#### Case-Control Studies

**Design Principle**:
```
Disease Status
Cases    Controls
  ↑        ↑
Look back in time
  ↓        ↓
Exposed  Unexposed
```

**Study Process**:
1. **Define cases**: Clear case definition, inclusion/exclusion criteria
2. **Select controls**: Representative of source population
3. **Assess exposure**: Retrospective exposure assessment
4. **Analyze data**: Calculate odds ratio

**Case Selection**:
- **Incident cases**: Newly diagnosed (preferred)
- **Prevalent cases**: Existing cases (if appropriate)
- **Source**: Hospitals, registries, surveillance systems
- **Verification**: Confirm diagnosis with medical records

**Control Selection**:
- **Population controls**: Random sample from general population
- **Hospital controls**: Patients with other conditions
- **Neighborhood controls**: Matched by residence
- **Friend/relative controls**: Minimize confounding

**Matching**:
- **Individual matching**: Each case matched to specific control(s)
- **Frequency matching**: Overall distributions similar
- **Variables**: Age, sex, time, place commonly matched
- **Analysis**: Must account for matching in analysis

**Advantages**:
- Efficient for rare diseases
- Can study multiple exposures
- Relatively quick and inexpensive
- Suitable for diseases with long latency

**Disadvantages**:
- Cannot calculate incidence
- Potential for recall bias
- Temporal relationship unclear
- Selection bias possible

**Example**: Lung cancer cases vs. controls, examining smoking history

#### Cohort Studies

**Design Principle**:
```
Exposure Status
Exposed  Unexposed
  ↓        ↓
Follow forward in time
  ↓        ↓
Disease  No Disease
```

**Types**:

**Prospective Cohort**:
- **Timeline**: Start with exposure, follow for outcomes
- **Data collection**: Primary data collection
- **Advantages**: High quality data, multiple outcomes
- **Disadvantages**: Expensive, long duration, loss to follow-up

**Retrospective Cohort**:
- **Timeline**: Historical cohort using existing records
- **Data collection**: Secondary data sources
- **Advantages**: Faster, less expensive
- **Disadvantages**: Limited data quality, incomplete records

**Study Process**:
1. **Define cohort**: Source population
2. **Assess exposure**: Baseline exposure measurement
3. **Follow-up**: Track participants over time
4. **Outcome ascertainment**: Systematic detection of outcomes
5. **Analysis**: Calculate incidence, relative risk

**Advantages**:
- Can calculate incidence rates
- Temporal relationship clear
- Multiple outcomes possible
- Less susceptible to bias

**Disadvantages**:
- Expensive and time-consuming
- Loss to follow-up
- Inefficient for rare diseases
- Exposure may change over time

**Example**: Framingham Heart Study following cardiovascular disease development

#### Nested Case-Control Studies

**Design Features**:
- **Cohort base**: Subset of larger cohort study
- **Case selection**: Cases arising within cohort
- **Control selection**: Random sample from cohort at risk
- **Exposure**: Often biological specimens stored at baseline

**Advantages**:
- Combines efficiency of case-control with cohort validity
- Reduces cost for expensive exposure measurements
- Temporal relationship clear
- Less selection bias

**Example**: Cancer cases within nurses' health study with stored blood samples

### Experimental Studies

#### Randomized Controlled Trials (RCTs)

**Design Features**:
```
Study Population
      ↓
Randomization
   ↙    ↘
Treatment  Control
Group      Group
   ↓        ↓
Outcome   Outcome
```

**Key Components**:
1. **Randomization**: Random allocation to groups
2. **Control group**: Comparison group (placebo/standard care)
3. **Blinding**: Participants/investigators unaware of allocation
4. **Follow-up**: Systematic outcome assessment

**Types of Blinding**:
- **Single-blind**: Participants blinded
- **Double-blind**: Participants and investigators blinded
- **Triple-blind**: Participants, investigators, and analysts blinded

**Randomization Methods**:
- **Simple randomization**: Coin flip, random number table
- **Block randomization**: Ensures balance within blocks
- **Stratified randomization**: Separate randomization by strata
- **Adaptive randomization**: Allocation changes based on results

**Advantages**:
- Strongest evidence for causation
- Controls for known and unknown confounders
- Can establish efficacy
- Minimizes bias

**Disadvantages**:
- Expensive and complex
- Ethical constraints
- May not reflect real-world effectiveness
- Limited generalizability

**Example**: COVID-19 vaccine trials comparing vaccine vs. placebo

#### Community Trials

**Design Features**:
- **Unit of randomization**: Communities or populations
- **Intervention**: Population-level intervention
- **Outcome**: Population health measures
- **Analysis**: Account for clustering

**Examples**:
- Water fluoridation trials
- Community cardiovascular disease prevention programs
- Mass vaccination campaigns

#### Quasi-experimental Studies

**Design Features**:
- **No randomization**: Assignment not random
- **Natural experiments**: Exposures occur naturally
- **Policy evaluations**: Before-after comparisons
- **Interrupted time series**: Changes in trends

**Examples**:
- Smoking ban effects on heart attacks
- Helmet law effects on head injuries
- School nutrition program evaluations

## Measures of Disease Frequency

### Fundamental Concepts

#### Counts vs. Rates
- **Count**: Absolute number of cases
- **Rate**: Count divided by population at risk
- **Importance**: Rates allow comparison between populations

#### Time Concepts
- **Point prevalence**: At specific point in time
- **Period prevalence**: Over specified time period
- **Incidence period**: Time during which new cases occur
- **Person-time**: Sum of time contributed by all individuals

### Prevalence Measures

#### Point Prevalence
**Formula**: 
```
Point Prevalence = Number of existing cases at time t / Total population at time t
```

**Interpretation**: Proportion of population with disease at specific time

**Example**: 
- 500 people with diabetes in town of 10,000 on January 1
- Point prevalence = 500/10,000 = 0.05 = 5%

#### Period Prevalence
**Formula**:
```
Period Prevalence = Number of cases existing during time period / Average population during time period
```

**Uses**: Chronic diseases, healthcare planning

#### Lifetime Prevalence
**Definition**: Proportion who have ever had the disease
**Uses**: Mental health conditions, chronic diseases

### Incidence Measures

#### Cumulative Incidence (Risk)
**Formula**:
```
Cumulative Incidence = Number of new cases during time period / Population at risk at beginning of period
```

**Assumptions**:
- Fixed population (no migration)
- All followed for complete period
- Competing risks minimal

**Interpretation**: Probability of developing disease during specified period

**Example**:
- 50 new cancer cases in 1 year
- Population at risk: 10,000 at start of year
- Cumulative incidence = 50/10,000 = 0.005 = 5 per 1,000 per year

#### Incidence Rate (Incidence Density)
**Formula**:
```
Incidence Rate = Number of new cases / Total person-time at risk
```

**Advantages**:
- Accounts for varying follow-up times
- Handles population changes
- No upper limit (theoretically)

**Person-time Calculation**:
- Sum of time each person contributes while at risk
- Stop counting when: develop disease, die, lost to follow-up, study ends

**Example**:
- 50 new cases
- 9,500 person-years of follow-up
- Incidence rate = 50/9,500 = 5.26 per 1,000 person-years

#### Attack Rate
**Definition**: Cumulative incidence during outbreak or epidemic
**Formula**: Cases during outbreak / Population exposed to risk
**Uses**: Outbreak investigations, foodborne illness

**Example**: 
- 25 people ill after church dinner
- 100 people attended dinner
- Attack rate = 25/100 = 25%

#### Case Fatality Rate
**Definition**: Proportion of cases who die from disease
**Formula**: Deaths from disease / Total cases of disease
**Time frame**: Usually within specified period

**Example**:
- 10 deaths from 100 COVID-19 cases in 30 days
- Case fatality rate = 10/100 = 10%

### Relationship Between Prevalence and Incidence

#### Steady-state Formula
**For stable conditions**:
```
Prevalence ≈ Incidence × Average Duration of Disease
```

**Factors affecting prevalence**:
- Incidence rate (↑ incidence → ↑ prevalence)
- Disease duration (↑ duration → ↑ prevalence)
- Case fatality (↑ fatality → ↓ prevalence)
- Cure rate (↑ cure → ↓ prevalence)
- Migration patterns

**Example Applications**:
- HIV: High incidence + long duration = high prevalence
- Influenza: High incidence + short duration = low prevalence
- Pancreatic cancer: Low incidence + short duration = low prevalence

## Measures of Association

### Measures in Cohort Studies

#### Relative Risk (Risk Ratio)
**Formula**:
```
RR = Risk in exposed group / Risk in unexposed group
```

**Calculation**:
```
        Disease    No Disease    Total
Exposed    a           b         a+b
Unexposed  c           d         c+d

Risk in exposed = a/(a+b)
Risk in unexposed = c/(c+d)
RR = [a/(a+b)] / [c/(c+d)]
```

**Interpretation**:
- RR = 1: No association
- RR > 1: Exposure increases risk
- RR < 1: Exposure decreases risk (protective)

**Example**:
- Smoking and lung cancer
- Risk in smokers: 150/1,000 = 0.15
- Risk in non-smokers: 5/1,000 = 0.005
- RR = 0.15/0.005 = 30

#### Rate Ratio (Incidence Rate Ratio)
**Formula**:
```
Rate Ratio = Incidence rate in exposed / Incidence rate in unexposed
```

**When to use**: When follow-up times vary between individuals

#### Risk Difference (Attributable Risk)
**Formula**:
```
RD = Risk in exposed - Risk in unexposed
```

**Interpretation**: Excess risk due to exposure
**Units**: Same as risk (proportion or rate)

**Example**:
- RD = 0.15 - 0.005 = 0.145
- 145 additional lung cancer cases per 1,000 smokers

#### Attributable Risk Percent
**Formula**:
```
AR% = (RR - 1) / RR × 100
```

**Interpretation**: Percentage of disease in exposed attributable to exposure

#### Population Attributable Risk
**Formula**:
```
PAR = Risk in total population - Risk in unexposed
```

**Interpretation**: Excess risk in population due to exposure

#### Population Attributable Risk Percent
**Formula**:
```
PAR% = [p(RR-1)] / [1 + p(RR-1)] × 100
```
Where p = proportion of population exposed

**Interpretation**: Percentage of disease in population attributable to exposure

### Measures in Case-Control Studies

#### Odds Ratio
**Formula**:
```
OR = (a×d) / (b×c)
```

**2×2 Table**:
```
           Cases   Controls
Exposed      a        b
Unexposed    c        d
```

**Interpretation**:
- OR = 1: No association
- OR > 1: Exposure increases odds of disease
- OR < 1: Exposure decreases odds of disease

**Relationship to RR**:
- When disease is rare (< 10%), OR ≈ RR
- OR always further from 1 than RR

**Example**:
- Lung cancer case-control study
- OR = (180×900)/(20×100) = 162,000/2,000 = 81

### Cross-sectional Studies

#### Prevalence Ratio
**Formula**:
```
PR = Prevalence in exposed / Prevalence in unexposed
```

#### Prevalence Odds Ratio
**Formula**: Same as odds ratio but using prevalent cases

### Confidence Intervals

#### Purpose
- Quantify uncertainty in estimates
- Provide range of plausible values
- Aid in interpretation of results

#### Calculation for Risk Ratio
**Formula**:
```
95% CI for ln(RR) = ln(RR) ± 1.96 × SE[ln(RR)]
```
Where:
```
SE[ln(RR)] = √[(1/a) - (1/(a+b)) + (1/c) - (1/(c+d))]
```

Then exponentiate limits to get CI for RR

#### Interpretation
- If CI includes 1: Not statistically significant at α=0.05
- If CI excludes 1: Statistically significant at α=0.05
- Width indicates precision

## Bias and Confounding

### Selection Bias

#### Definition
Systematic error in selection of study participants or classification of participants into study groups

#### Types in Case-Control Studies

**Berkson's Bias (Admission Rate Bias)**:
- **Mechanism**: Cases and controls selected from different populations
- **Example**: Hospital controls may have different exposure patterns than general population
- **Prevention**: Use population-based controls when possible

**Neyman Bias (Prevalence-Incidence Bias)**:
- **Mechanism**: Using prevalent rather than incident cases
- **Problem**: Survival affects case selection
- **Prevention**: Use incident cases when studying causation

**Control Selection Bias**:
- **Mechanism**: Controls not representative of source population
- **Examples**: Volunteer bias, non-response bias
- **Prevention**: Maximize response rates, assess non-response

#### Types in Cohort Studies

**Healthy Worker Effect**:
- **Mechanism**: Workers healthier than general population
- **Result**: Underestimation of occupational health risks
- **Example**: Comparing worker mortality to general population

**Loss to Follow-up Bias**:
- **Mechanism**: Differential loss related to exposure and outcome
- **Prevention**: Minimize loss, assess characteristics of lost participants

### Information Bias (Measurement Bias)

#### Definition
Systematic error in measurement or classification of exposure or outcome

#### Recall Bias
**Mechanism**: Cases remember exposures differently than controls
**Factors affecting recall**:
- Severity of disease
- Publicity about exposure-disease association
- Time since exposure

**Prevention strategies**:
- Use biological markers when possible
- Blind interviewers to case status
- Use structured interviews
- Validate self-reported information

**Example**: Cancer patients may recall pesticide exposure more completely

#### Interviewer Bias
**Mechanism**: Interviewer behavior differs by case status
**Prevention**:
- Standardized protocols
- Blind interviewers
- Training and monitoring

#### Misclassification Bias

**Non-differential Misclassification**:
- **Definition**: Misclassification rate same in compared groups
- **Effect**: Generally biases toward null (OR/RR toward 1)
- **Example**: Random measurement error

**Differential Misclassification**:
- **Definition**: Misclassification rate differs between groups
- **Effect**: Can bias toward or away from null
- **Example**: Cases report exposure more accurately

### Confounding

#### Definition
Mixing of the effect of exposure with effect of another variable (confounder)

#### Criteria for Confounding Variable
1. **Associated with exposure**: Must be related to exposure in study population
2. **Independent risk factor**: Must cause outcome independent of exposure
3. **Not in causal pathway**: Must not be intermediate step between exposure and outcome

#### Confounding Examples

**Age as Confounder**:
- Exposure: Physical activity
- Outcome: Heart disease
- Confounder: Age (older people less active, higher heart disease risk)

**Smoking as Confounder**:
- Exposure: Alcohol consumption
- Outcome: Lung cancer
- Confounder: Smoking (correlated with drinking, causes lung cancer)

#### Detection of Confounding

**Crude vs. Adjusted Measures**:
- Calculate crude association
- Calculate stratified/adjusted association
- If substantial difference: confounding present

**10% Rule**:
If adjusted estimate differs from crude by ≥10%, consider confounding important

#### Control of Confounding

##### Design Phase

**Randomization** (Experimental Studies):
- **Mechanism**: Random allocation balances known/unknown confounders
- **Most effective**: Gold standard for confounder control
- **Limitation**: Only feasible in experimental studies

**Restriction**:
- **Mechanism**: Limit study to homogeneous group
- **Example**: Study only non-smokers to eliminate smoking as confounder
- **Disadvantages**: Limits generalizability, reduces sample size

**Matching**:
- **Individual matching**: Each case matched to control(s) on confounders
- **Frequency matching**: Overall distributions balanced
- **Variables**: Usually age, sex, race
- **Analysis**: Must use matched analysis methods

##### Analysis Phase

**Stratification (Mantel-Haenszel)**:
- **Process**: Calculate stratum-specific estimates, combine with weights
- **Advantage**: Shows effect modification
- **Limitation**: Limited to few variables with few categories

**Multivariable Analysis**:
- **Logistic regression**: For case-control studies
- **Cox regression**: For cohort studies with time-to-event
- **Linear regression**: For continuous outcomes
- **Advantage**: Can control for many variables simultaneously

#### Effect Modification (Interaction)

#### Definition
When effect of exposure on outcome differs across levels of third variable

#### Difference from Confounding
- **Confounding**: Nuisance to be controlled
- **Effect modification**: Biological phenomenon to be described

#### Example
- Asbestos exposure and lung cancer
- Effect much stronger in smokers than non-smokers
- Smoking is effect modifier

#### Assessment
- **Statistical tests**: Test for interaction terms
- **Stratified analysis**: Compare stratum-specific estimates
- **Biological plausibility**: Consider mechanism

## Causal Inference

### Hill's Criteria for Causation

Developed by Austin Bradford Hill (1965) for evaluating causation from observational studies:

#### 1. Strength of Association
- **Strong associations**: More likely to be causal
- **Weak associations**: May be due to bias/confounding
- **Example**: RR=30 for smoking and lung cancer suggests causation

#### 2. Consistency
- **Replication**: Results consistent across different studies
- **Populations**: Consistent in different populations
- **Methods**: Consistent with different study designs

#### 3. Temporal Relationship
- **Sequence**: Exposure must precede disease
- **Critical**: Necessary for causation
- **Challenge**: Establishing timing in cross-sectional studies

#### 4. Biological Gradient (Dose-Response)
- **Pattern**: Increasing exposure → increasing risk
- **Types**: Linear, threshold, J-shaped
- **Example**: More cigarettes per day → higher lung cancer risk

#### 5. Biological Plausibility
- **Mechanism**: Scientifically reasonable mechanism
- **Knowledge dependent**: Limited by current understanding
- **Example**: Carcinogens causing cancer

#### 6. Coherence
- **Consistency**: Doesn't conflict with natural history/biology
- **Compatibility**: Fits with other knowledge
- **Example**: Lung cancer trends match smoking trends

#### 7. Experimental Evidence
- **Intervention studies**: Removing exposure reduces disease
- **Animal studies**: Laboratory evidence
- **Example**: Smoking cessation reduces lung cancer risk

#### 8. Analogy
- **Similar exposures**: Analogous exposures cause similar effects
- **Example**: If one drug causes birth defects, similar drugs might too

#### 9. Specificity
- **One cause-one effect**: Single exposure causes single disease
- **Least important**: Many diseases have multiple causes
- **Modern view**: Less emphasis on this criterion

### Modern Causal Inference

#### Counterfactual Framework

**Definition**: What would have happened to same individuals if exposure status had been different

**Fundamental Problem**: Cannot observe both exposed and unexposed outcomes in same person

**Causal Effect**:
Individual level: Y₁ᵢ - Y₀ᵢ (unobservable)
Population level: E[Y₁] - E[Y₀] (estimable under assumptions)

#### Directed Acyclic Graphs (DAGs)

**Purpose**: Graphical representation of causal assumptions

**Components**:
- **Nodes**: Variables
- **Arrows**: Causal relationships
- **Paths**: Sequences of arrows

**Types of Paths**:
- **Causal paths**: Exposure → Outcome
- **Backdoor paths**: Confounding paths
- **Collider paths**: Paths through colliders (A → C ← B)

**Example DAG**:
```
Smoking → Lung Cancer
    ↑           ↑
    Age --------+
```

#### Causal Assumptions

**Exchangeability (No Unmeasured Confounding)**:
- Exposed and unexposed groups comparable except for exposure
- No unmeasured confounders

**Positivity**:
- All individuals have non-zero probability of each exposure level
- Prevents extrapolation beyond data

**Consistency (SUTVA)**:
- Same exposure produces same outcome
- No interference between individuals

**No Measurement Error**:
- Exposure and outcome measured without error

#### Propensity Score Methods

**Propensity Score**: Probability of receiving treatment given observed covariates

**Uses**:
- **Matching**: Match on propensity score
- **Stratification**: Stratify by propensity score quintiles  
- **Weighting**: Weight by inverse probability of treatment
- **Covariate adjustment**: Include propensity score in model

**Advantages**:
- Reduces dimensionality
- Explicit about treatment assignment mechanism
- Can assess overlap between groups

**Limitations**:
- Still assumes no unmeasured confounding
- Requires good propensity score model

#### Instrumental Variables

**Definition**: Variable that affects exposure but not outcome except through exposure

**Assumptions**:
1. **Relevance**: Instrument associated with exposure
2. **Independence**: Instrument independent of unmeasured confounders
3. **Exclusion restriction**: Instrument affects outcome only through exposure

**Examples**:
- **Mendelian randomization**: Genetic variants as instruments
- **Policy instruments**: Changes in laws/policies
- **Geographic variation**: Distance to healthcare facility

**Advantages**:
- Can handle unmeasured confounding
- Identifies causal effects under assumptions

**Limitations**:
- Strong assumptions required
- Often weak instruments (low power)
- Local average treatment effect interpretation

This completes the first comprehensive section on epidemiology fundamentals. The content covers the essential theoretical foundation needed for data science applications in epidemiology. Would you like me to continue with the next section covering R programming for epidemiology, including detailed RStudio setup, project organization, and best practices? 