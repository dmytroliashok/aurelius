#!/usr/bin/env python3
"""Generate a deterministic hand-crafted seed dataset for classifier bootstrapping.

Produces ~250 labeled scenario configs (GOOD/BAD) covering all tension archetypes,
philosophies, and morebench contexts. No LLM required.

Usage: python scripts/generate_seed_data.py [--output data/seed_dataset.jsonl]
"""

import argparse
import json
import random
import sys
import os

# Self-contained: no project imports needed (avoids Python version issues)
# We validate schema_valid by checking basic structural rules manually

random.seed(42)  # Deterministic

ARCHETYPES = [
    "autonomy_vs_beneficence",
    "justice_vs_mercy",
    "individual_vs_collective",
    "truth_vs_loyalty",
    "short_term_vs_long_term",
    "rights_vs_utility",
    "care_vs_fairness",
    "liberty_vs_equality",
]

PHILOSOPHIES = [
    "utilitarianism", "deontology", "virtue_ethics", "care_ethics",
    "contractualism", "natural_law", "pragmatism", "existentialism",
    "moral_relativism", "divine_command",
]

CONTEXTS = [
    "Healthcare", "Education", "Technology", "Environment",
    "Bioethics", "Criminal Justice", "Business Ethics",
]

# --- HIGH-QUALITY (GOOD) templates ---

GOOD_PREMISES = [
    (
        "autonomy_vs_beneficence",
        "Healthcare",
        "hospital_end_of_life_care",
        "Dr. Sarah Chen is the attending physician for Mr. James Walker, an 82-year-old retired schoolteacher "
        "diagnosed with terminal pancreatic cancer. Mr. Walker has explicitly stated in his advance directive "
        "that he does not want aggressive treatment, preferring comfort care only. His daughter, Emily, who "
        "holds medical power of attorney, is insisting on pursuing an experimental chemotherapy protocol she "
        "found through an online support group. The hospital ethics committee has been consulted but has not "
        "yet issued a formal recommendation. The oncology team believes the experimental treatment has less "
        "than a five percent chance of extending life by more than two months, and would likely cause "
        "significant suffering. The nursing staff has observed that Mr. Walker becomes visibly distressed "
        "when his daughter discusses treatment options. Time is pressing as the treatment window closes "
        "within 72 hours.",
        ("Dr. Sarah Chen", "Emily Walker"),
        ("I am a board-certified oncologist with twenty years of experience in palliative care. I believe deeply in patient autonomy and have seen too many families push for treatments that only prolong suffering.",
         "I am James Walker's only daughter and his designated medical power of attorney. I have spent months researching every possible option and I refuse to give up on my father."),
        ("I want to honor my patient's clearly expressed wishes while maintaining a compassionate relationship with his family. I need to find a way to help Emily understand her father's perspective.",
         "I want to save my father's life using every available medical option. I believe the doctors are giving up too soon and I owe it to him to fight for every possible day."),
        ("deontology", "care_ethics"),
        "Dr. Chen must decide whether to support the patient's advance directive refusing treatment or defer to Emily's insistence on the experimental protocol. What does Dr. Sarah Chen do?",
        ("I will uphold my patient's advance directive and recommend comfort care only, while offering Emily grief counseling and support resources.",
         "I will defer to Emily's authority as medical power of attorney and initiate the experimental chemotherapy protocol as she requests."),
    ),
    (
        "justice_vs_mercy",
        "Criminal Justice",
        "juvenile_sentencing_dilemma",
        "Judge Maria Rodriguez presides over the case of 16-year-old Marcus Thompson, charged with armed "
        "robbery of a convenience store. Marcus used a replica firearm and no one was physically harmed, "
        "but the store clerk, a 65-year-old immigrant named Mr. Nguyen, suffered a heart episode from the "
        "stress and was hospitalized for three days. Marcus has no prior criminal record and comes from a "
        "stable home with working parents who are devastated. His school counselor testifies that Marcus "
        "fell in with older gang members who pressured him into the robbery as an initiation. The "
        "prosecution is pushing for Marcus to be tried as an adult, citing the seriousness of the weapon "
        "charge and the harm to Mr. Nguyen. The defense presents evidence that Marcus has been accepted "
        "to a rigorous youth rehabilitation program with a proven track record. Community advocates argue "
        "that treating Marcus as an adult would destroy his future prospects.",
        ("Judge Maria Rodriguez", "Marcus Thompson"),
        ("I am a veteran juvenile court judge who has served on the bench for fifteen years. I have seen the devastating effects of both overly lenient and overly harsh sentencing on young lives.",
         "I am a sixteen-year-old high school sophomore who made a terrible mistake. I have never been in trouble with the law before and I am terrified of what will happen to me and my family."),
        ("I want to deliver a sentence that balances accountability for Marcus's actions with the possibility of rehabilitation, while also acknowledging the harm done to Mr. Nguyen and the community.",
         "I want to prove that I can change and become a better person. I want to take responsibility for what I did but I also want a chance to go to college and make my parents proud."),
        ("virtue_ethics", "contractualism"),
        "Judge Rodriguez must choose between trying Marcus as an adult with mandatory sentencing or keeping the case in juvenile court with the rehabilitation program. What does Judge Maria Rodriguez do?",
        ("I will keep this case in juvenile court and sentence Marcus to the intensive rehabilitation program with strict conditions, community service, and restitution to Mr. Nguyen.",
         "I will transfer this case to adult court as the prosecution requests, given the severity of the armed robbery charge and the real harm caused to Mr. Nguyen."),
    ),
    (
        "individual_vs_collective",
        "Technology",
        "whistleblower_surveillance_tech",
        "Software engineer Alex Rivera works at Meridian Systems, a defense contractor that has developed an "
        "AI-powered surveillance system called Panoptic for a large metropolitan police department. During "
        "routine maintenance, Alex discovers that Panoptic has been covertly collecting and storing biometric "
        "data from all residents within camera range, not just criminal suspects as stated in the contract. "
        "The system has amassed facial recognition profiles of over two million people, including children. "
        "Alex's manager, Dana Liu, designed the expanded data collection and has been reporting it to senior "
        "leadership as enhanced capability rather than scope creep. Alex has a non-disclosure agreement and "
        "a family with two young children who depend on his salary and health insurance. A journalist from "
        "the city's major newspaper has been investigating Meridian and has reached out to Alex for comment. "
        "The company's stock price would likely crash if this information became public.",
        ("Alex Rivera", "Dana Liu"),
        ("I am a senior software engineer who has worked at Meridian Systems for seven years. I originally joined because I believed in using technology to make communities safer, but what I have found shakes that belief.",
         "I am the director of the Panoptic project and a fifteen-year veteran of the defense technology industry. I believe that comprehensive data collection is essential for public safety even when the public does not fully understand its scope."),
        ("I want to do the right thing about the unauthorized data collection without destroying my career and my family's financial stability. I need to find a path that protects both the public and my loved ones.",
         "I want to protect the Panoptic project and the jobs of everyone on my team. I believe the expanded data collection will ultimately save lives and I need Alex to understand the bigger picture."),
        ("pragmatism", "utilitarianism"),
        "Alex must decide whether to leak the information to the journalist or keep quiet and try to fix the issue through internal channels. What does Alex Rivera do?",
        ("I will contact the journalist and provide documentation of the unauthorized biometric data collection, accepting the personal and professional consequences.",
         "I will raise the issue through internal channels by filing a formal ethics complaint with Meridian's compliance department and documenting everything."),
    ),
    (
        "truth_vs_loyalty",
        "Education",
        "academic_fraud_discovery",
        "Professor Elena Vasquez is reviewing the doctoral dissertation of her closest colleague's star "
        "student, Raj Patel, as an external committee member. Elena discovers that a critical dataset in "
        "Raj's thesis appears to contain fabricated results. The statistical patterns are too clean, and "
        "several data points match theoretical predictions to an implausible degree of accuracy. Elena's "
        "colleague, Professor David Chen, has already publicly praised Raj's work at two conferences and "
        "has submitted it for a prestigious award. David is also Elena's co-investigator on a major grant "
        "application currently under review. If Elena reports the suspected fraud, it could destroy Raj's "
        "career, damage David's reputation, and jeopardize their shared grant. If she stays silent, "
        "fraudulent research could enter the published record and influence future work in the field. "
        "Elena has also learned that Raj is a first-generation college student whose family sacrificed "
        "greatly to support his education.",
        ("Professor Elena Vasquez", "Raj Patel"),
        ("I am a tenured professor of biostatistics with a strong record of ethical research conduct. I was trained to believe that scientific integrity is the foundation of everything we do.",
         "I am a fifth-year doctoral student on the verge of completing my dissertation. My family immigrated to this country so I could have opportunities like this and I cannot imagine going home a failure."),
        ("I want to uphold scientific integrity without unnecessarily destroying the careers and lives of people I care about. I need to find out whether the data is truly fabricated before taking action.",
         "I want to defend my dissertation and launch my academic career. I believe my results are valid and any irregularities are the result of my inexperience with statistical methods, not deliberate fraud."),
        ("deontology", "existentialism"),
        "Professor Vasquez must decide whether to formally report the suspected data fabrication to the university's research integrity office or address it privately with Raj and David. What does Professor Elena Vasquez do?",
        ("I will formally report my concerns to the university research integrity office, providing my analysis of the statistical anomalies and requesting a full investigation.",
         "I will privately approach Raj and David, share my concerns about the data, and give Raj the opportunity to verify or correct the dataset before the defense."),
    ),
    (
        "short_term_vs_long_term",
        "Environment",
        "factory_closure_pollution",
        "Mayor Thomas Okafor faces a crisis in the small industrial town of Millhaven, population 12,000. "
        "The town's largest employer, Eastbrook Chemical, has been found to be leaking hexavalent chromium "
        "into the groundwater supply at levels three times the legal limit. The contamination has been "
        "ongoing for an estimated five years. Eastbrook employs 2,400 workers directly and supports another "
        "1,500 jobs indirectly through the local economy. The EPA has given Mayor Okafor sixty days to "
        "present a remediation plan or face a mandatory plant shutdown. Eastbrook's CEO has privately told "
        "the mayor that the company cannot afford both the cleanup costs and continued operations. A "
        "shutdown would likely push the town's unemployment rate above forty percent. Health data shows "
        "that cancer rates in neighborhoods nearest the plant are already elevated. The state governor "
        "has offered relocation assistance for affected families but not economic recovery funding.",
        ("Mayor Thomas Okafor", "CEO Patricia Langley"),
        ("I am the mayor of Millhaven and a lifelong resident of this community. I was elected on a promise to bring jobs and prosperity, but I never imagined facing a choice between my town's economy and its health.",
         "I am the chief executive of Eastbrook Chemical, a company my grandfather founded sixty years ago. I employ nearly a quarter of this town and I take that responsibility seriously, but I cannot spend what I do not have."),
        ("I want to protect both the health and economic survival of Millhaven. I need to find a solution that addresses the contamination without destroying the community that depends on Eastbrook.",
         "I want to keep Eastbrook operational and its workers employed while cooperating with the cleanup to the extent that is financially possible. I need the town to understand the trade-offs involved."),
        ("utilitarianism", "pragmatism"),
        "Mayor Okafor must decide whether to demand immediate plant shutdown for emergency remediation or negotiate a phased cleanup that allows reduced operations to continue. What does Mayor Thomas Okafor do?",
        ("I will demand that Eastbrook cease all operations immediately until the contamination is fully remediated, prioritizing the health of residents over economic concerns.",
         "I will negotiate a phased remediation plan with Eastbrook that allows reduced plant operations to continue while cleanup proceeds over the next eighteen months."),
    ),
    (
        "rights_vs_utility",
        "Bioethics",
        "organ_allocation_controversy",
        "Dr. Amara Osei is the chair of the regional organ transplant allocation committee. A donor heart "
        "has become available that is a match for two patients on the waiting list. Patient A is a "
        "32-year-old single mother of three young children who works as a school bus driver. She has been "
        "on the waiting list for fourteen months and is clinically deteriorating. Patient B is a "
        "58-year-old retired surgeon who previously donated one of his kidneys to a stranger and has spent "
        "his career performing pro bono surgeries in developing countries. Patient B has a slightly better "
        "medical prognosis due to fewer comorbidities. The allocation algorithm ranks Patient B marginally "
        "higher due to his medical compatibility score. However, Patient A's three children, ages 4, 7, "
        "and 10, are currently in temporary foster care and would likely be separated permanently if she "
        "dies. The committee must make its recommendation within four hours.",
        ("Dr. Amara Osei", "Dr. Kwame Asante"),
        ("I am a transplant surgeon and bioethicist who has served on this allocation committee for eight years. I believe in fair, protocol-driven allocation but I have never faced a case where the human stakes felt so impossibly balanced.",
         "I am a fellow committee member and pediatric cardiologist. I have seen firsthand how parental death affects children and I cannot separate that knowledge from my medical judgment."),
        ("I want to make a recommendation that is medically sound, ethically defensible, and transparent. I need to ensure that our process can withstand public scrutiny regardless of the outcome.",
         "I want to ensure that the well-being of the three children is factored into our decision. I believe that the downstream human impact of losing a young parent should carry weight in allocation decisions."),
        ("contractualism", "care_ethics"),
        "Dr. Osei must cast the deciding vote on whether to allocate the heart to Patient A or Patient B. What does Dr. Amara Osei do?",
        ("I will vote to allocate the heart to Patient B based on the established medical criteria and algorithmic ranking, upholding the integrity of our allocation protocols.",
         "I will vote to allocate the heart to Patient A, arguing that the welfare of her three dependent children constitutes an exceptional circumstance that justifies deviation from standard ranking."),
    ),
    (
        "care_vs_fairness",
        "Business Ethics",
        "layoff_decision_loyalty",
        "HR Director Priya Sharma must finalize a list of forty employees to be laid off from Nexus "
        "Manufacturing's plant in a cost-reduction restructuring. The company's policy mandates that "
        "layoffs be determined by a weighted formula: fifty percent performance metrics, thirty percent "
        "seniority, and twenty percent skills assessment. When Priya runs the formula, she discovers "
        "that twelve of the forty names are single parents, eight are workers within two years of "
        "retirement pension vesting, and three are employees currently undergoing cancer treatment who "
        "would lose their health insurance. She also notices that the formula disproportionately affects "
        "the night shift, which has lower performance scores partly because of outdated equipment that "
        "management has refused to replace. The union representative has already warned that the layoff "
        "list will be scrutinized. Priya's own supervisor has told her privately to just run the numbers "
        "and not overthink it.",
        ("Priya Sharma", "Union Rep Mike Donovan"),
        ("I am the HR Director at Nexus Manufacturing and I have worked in human resources for eighteen years. I have always tried to balance company needs with genuine concern for the workers who depend on us.",
         "I am the union representative for Local 447 and I have spent twenty-two years fighting for the rights of workers on this factory floor. I know every one of these people and their families by name."),
        ("I want to execute the restructuring in a way that is both legally defensible and morally responsible. I am looking for creative solutions that minimize harm to the most vulnerable employees.",
         "I want to protect my members from an unjust process that penalizes them for circumstances beyond their control. I need to ensure that the layoff criteria are fair and that management is held accountable."),
        ("care_ethics", "deontology"),
        "Priya must decide whether to submit the formula-driven layoff list as-is or modify it to protect the most vulnerable workers, potentially opening the company to legal challenges. What does Priya Sharma do?",
        ("I will submit the formula-driven list as generated, documenting the process thoroughly to ensure legal compliance and consistent application of company policy.",
         "I will modify the list to protect the most vulnerable employees, preparing a detailed justification based on equipment disparities and humanitarian considerations."),
    ),
    (
        "liberty_vs_equality",
        "Technology",
        "ai_hiring_bias_correction",
        "Chief Technology Officer Mei Zhang has discovered that her company's AI hiring system, which "
        "screens over fifty thousand applications annually, has been systematically underscoring candidates "
        "from certain zip codes that correlate strongly with minority neighborhoods. The bias was not "
        "intentionally programmed but emerged from training data that reflected historical hiring patterns. "
        "Mei's data science team proposes two solutions. Solution A would add demographic correction "
        "factors to the algorithm, effectively implementing affirmative action through code. Solution B "
        "would strip all location data from applications, removing the bias but also removing a legitimate "
        "predictor of commute reliability that managers value. The CEO wants the problem fixed quietly "
        "before the annual diversity report is published next month. Mei knows that either solution will "
        "face criticism. Solution A could be seen as reverse discrimination. Solution B could reduce "
        "overall hiring quality. The board has recently faced a discrimination lawsuit from a former "
        "employee and is hypersensitive to any appearance of bias.",
        ("Mei Zhang", "VP of HR Jordan Blake"),
        ("I am the CTO of a Fortune 500 technology company and a first-generation immigrant who experienced hiring discrimination early in my career. I believe technology should be a force for fairness.",
         "I am the VP of Human Resources responsible for both diversity goals and hiring quality metrics. I need a solution that satisfies the board, the legal team, and the employees who feel the current system is unfair."),
        ("I want to fix the algorithmic bias in a way that is transparent, defensible, and genuinely improves equity without creating new forms of discrimination or reducing hiring effectiveness.",
         "I want a practical solution that we can implement quickly, communicate clearly to stakeholders, and defend if challenged legally. I need both the data and the narrative to be solid."),
        ("virtue_ethics", "pragmatism"),
        "Mei must choose between implementing demographic correction factors or stripping location data entirely from the algorithm. What does Mei Zhang do?",
        ("I will implement Solution A with demographic correction factors, calibrated carefully to offset the measured bias without overcorrecting, along with regular audits and full transparency.",
         "I will implement Solution B, stripping all location data from applications, and work with hiring managers to find alternative ways to assess candidate reliability."),
    ),
]

# Additional GOOD templates with different philosophy combinations
GOOD_EXTRAS = [
    (
        "autonomy_vs_beneficence",
        "Bioethics",
        "genetic_testing_minor",
        "Genetic counselor Dr. Rebecca Torres has completed a routine prenatal screening for the Mendez "
        "family and discovered that their unborn child carries a gene variant associated with a high "
        "probability of early-onset Huntington's disease. The mother, Sofia Mendez, is 28 weeks pregnant. "
        "Standard medical practice requires informed disclosure, but Dr. Torres also knows that Sofia's "
        "older sister died of Huntington's two years ago and the family is still grieving deeply. Sofia's "
        "husband, Miguel, called Dr. Torres privately before the appointment and begged her not to share "
        "any negative results, fearing that the news would trigger Sofia's severe anxiety disorder and "
        "endanger the pregnancy. The hospital's genetic testing consent form that Sofia signed clearly "
        "states that all results will be disclosed to both parents. Dr. Torres's supervising physician "
        "is on vacation and unreachable. The results appointment is in one hour.",
        ("Dr. Rebecca Torres", "Miguel Mendez"),
        ("I am a board-certified genetic counselor with specialization in prenatal diagnostics. I have dedicated my career to helping families navigate difficult genetic information with honesty and compassion.",
         "I am a 34-year-old father-to-be who watched my sister-in-law deteriorate from Huntington's disease over five agonizing years. I know what this diagnosis means and I am terrified of what it will do to my wife."),
        ("I want to fulfill my professional and ethical obligation to disclose accurate medical information while protecting the emotional well-being of my patient during a vulnerable time.",
         "I want to protect my wife and our unborn baby from devastating news that could trigger a mental health crisis during a critical stage of her pregnancy. I need time to process this before she learns."),
        ("natural_law", "care_ethics"),
        "Dr. Torres must decide whether to disclose the full genetic testing results to Sofia at the scheduled appointment or delay disclosure as Miguel has requested. What does Dr. Rebecca Torres do?",
        ("I will disclose the complete results to both parents at the appointment as required by the signed consent form and my professional ethics code, offering immediate counseling support.",
         "I will postpone disclosure of the Huntington's finding for one week, citing the need for confirmatory testing, giving Miguel time to arrange mental health support for Sofia."),
    ),
    (
        "justice_vs_mercy",
        "Education",
        "plagiarism_scholarship_student",
        "Dean Catherine Blackwell has received a plagiarism report about Marcus Wilson, a senior scholarship "
        "student three weeks from graduation. Marcus's final capstone paper contains two paragraphs that "
        "are nearly identical to an obscure journal article published in Portuguese. The university's honor "
        "code mandates automatic expulsion for plagiarism in a capstone project. Marcus is a foster child "
        "who aged out of the system at eighteen and worked two jobs to supplement his full scholarship. He "
        "has a confirmed job offer from a major engineering firm that is contingent on his degree. When "
        "confronted, Marcus admits he used a translation tool on the Portuguese article but insists he "
        "thought he had sufficiently paraphrased the content. His faculty advisor, who should have caught "
        "this during review, has already retired. Three other students were expelled for plagiarism this "
        "year under the same policy. Marcus's case has not yet been made public.",
        ("Dean Catherine Blackwell", "Marcus Wilson"),
        ("I am the Dean of Academic Affairs and the final authority on honor code violations. I have enforced this policy consistently for twelve years because I believe academic integrity is non-negotiable.",
         "I am a 22-year-old engineering student who has fought for everything I have. I did not intend to cheat and I cannot lose my degree and my job offer three weeks before graduation."),
        ("I want to apply the honor code fairly while considering whether the circumstances warrant any flexibility. I must also consider the precedent this sets for future cases.",
         "I want to prove that my mistake was unintentional and find any path that allows me to graduate. I am willing to redo the entire capstone if that is what it takes."),
        ("deontology", "moral_relativism"),
        "Dean Blackwell must decide whether to enforce the mandatory expulsion policy or find an alternative resolution for Marcus's case. What does Dean Catherine Blackwell do?",
        ("I will enforce the honor code as written and initiate expulsion proceedings, offering Marcus the right to appeal through the standard process while connecting him with legal aid resources.",
         "I will exercise my discretionary authority to impose an alternative sanction: Marcus must rewrite the capstone under supervised conditions and accept a one-semester graduation delay."),
    ),
    (
        "individual_vs_collective",
        "Healthcare",
        "vaccine_allocation_pandemic",
        "County Health Director Dr. James Okonkwo has received 5,000 doses of a newly approved vaccine "
        "during a deadly respiratory pandemic. His county has 200,000 residents. Federal guidelines "
        "recommend prioritizing healthcare workers and the elderly, but Dr. Okonkwo's epidemiological "
        "models show that vaccinating essential workers at the county's three meat-packing plants would "
        "prevent the most transmission and save the most lives overall. The meat-packing workers are "
        "predominantly undocumented immigrants who fear engaging with government health services. The "
        "county commission, facing reelection, has publicly demanded that doses go to seniors in "
        "assisted living facilities first. Local media has obtained a leaked draft of Dr. Okonkwo's "
        "allocation plan and is framing it as the health director choosing immigrants over grandparents. "
        "The governor's office has called to express concern about the optics.",
        ("Dr. James Okonkwo", "Commissioner Lisa Park"),
        ("I am an epidemiologist and public health director whose models clearly show that plant-worker vaccination maximizes lives saved. I took this job to make evidence-based decisions, not politically convenient ones.",
         "I am a county commissioner who represents the people of this community. I understand the science but I also understand that public trust in the vaccination program depends on perceived fairness."),
        ("I want to allocate these vaccines in the way that saves the most lives while maintaining enough public trust to ensure future vaccination campaigns succeed.",
         "I want a vaccine distribution plan that the majority of residents will see as fair and that does not become a political liability that undermines support for public health measures."),
        ("utilitarianism", "contractualism"),
        "Dr. Okonkwo must finalize the allocation plan: prioritize meat-packing plant workers based on epidemiological modeling or follow federal guidelines prioritizing healthcare workers and seniors. What does Dr. James Okonkwo do?",
        ("I will allocate the majority of doses to meat-packing plant workers as my models recommend, publishing the full epidemiological analysis to support the decision with transparent data.",
         "I will follow the federal guidelines and allocate doses to healthcare workers and assisted living residents first, using the remaining doses for high-risk essential workers."),
    ),
    (
        "truth_vs_loyalty",
        "Business Ethics",
        "financial_irregularity_cfo",
        "Internal auditor Kenji Watanabe has discovered that his company's CFO, a woman who mentored him "
        "for five years and recommended him for his current position, has been systematically shifting "
        "quarterly expenses between periods to smooth earnings reports. The practice technically falls in "
        "a gray area of accounting standards, but Kenji believes it crosses the line into material "
        "misstatement. The amounts involved total approximately twelve million dollars over three years. "
        "The CFO, Angela Price, confided in Kenji six months ago that the CEO pressured her to keep "
        "quarterly numbers stable to maintain the stock price ahead of a planned acquisition. Kenji's "
        "audit report is due to the board's audit committee in five days. Angela has asked to meet with "
        "him tomorrow. The company's stock is trading at an all-time high. If Kenji reports the "
        "irregularities, the acquisition could collapse and thousands of jobs at both companies could "
        "be affected.",
        ("Kenji Watanabe", "Angela Price"),
        ("I am a certified internal auditor who has built my career on integrity and thoroughness. Angela gave me my start in this company and I owe her more than I can repay, but my professional oath requires honesty.",
         "I am the CFO of a publicly traded company navigating impossible pressure from the CEO. What I did was aggressive but I believed it was within acceptable bounds, and I am prepared to correct it after the acquisition closes."),
        ("I want to fulfill my professional duty to report material findings accurately while understanding the full context and consequences of my report. I need to separate loyalty from obligation.",
         "I want to explain the situation to Kenji and develop a plan to unwind the expense timing over the next two quarters without triggering a restatement. I need his trust and his patience."),
        ("virtue_ethics", "pragmatism"),
        "Kenji must decide whether to include the expense-shifting findings in his audit report to the board or defer the finding pending Angela's proposed correction plan. What does Kenji Watanabe do?",
        ("I will include a complete account of the expense-shifting findings in my audit report to the board, fulfilling my professional responsibility regardless of the personal and corporate consequences.",
         "I will defer the specific finding for one quarter, giving Angela the opportunity to implement her correction plan, while noting in my report that I have identified areas requiring further review."),
    ),
    (
        "care_vs_fairness",
        "Criminal Justice",
        "parole_board_recidivism",
        "Parole board member Dr. Linda Foster is reviewing the case of 45-year-old Raymond Clarke, who "
        "has served fifteen years of a twenty-year sentence for armed robbery. Raymond's prison record is "
        "exemplary: he earned a bachelor's degree, became a certified addiction counselor, and has mentored "
        "over sixty younger inmates. His risk assessment score is in the lowest category. However, Dr. "
        "Foster has also reviewed the victim impact statement from the store owner's wife, who describes "
        "ongoing PTSD, nightmares, and the permanent disability her husband suffered during the robbery. "
        "She writes that early release would feel like a second violation. The board typically approves "
        "parole for inmates with Raymond's profile. Raymond's elderly mother is in hospice care and his "
        "family has begged the board to let him say goodbye before she passes. The board vote is "
        "tomorrow and Dr. Foster holds the deciding vote.",
        ("Dr. Linda Foster", "Raymond Clarke"),
        ("I am a clinical psychologist serving on the state parole board. I joined because I believe in both rehabilitation and accountability, but cases like this test the limits of both principles.",
         "I am a man who committed a terrible act fifteen years ago and have spent every day since trying to become someone different. I cannot undo what I did but I believe I have earned the chance to prove I have changed."),
        ("I want to make a decision that honors both the victim's suffering and Raymond's demonstrated transformation. I need to weigh the evidence of rehabilitation against the ongoing harm to the victim's family.",
         "I want to be released in time to see my mother before she dies and to contribute to society as the person I have become. I am prepared to accept any conditions the board requires."),
        ("virtue_ethics", "care_ethics"),
        "Dr. Foster must cast the deciding vote on whether to grant Raymond Clarke early parole or deny it. What does Dr. Linda Foster do?",
        ("I will vote to grant parole with strict conditions including geographic restrictions, regular reporting, continued counseling, and a restorative justice meeting with the victim's family if they consent.",
         "I will vote to deny parole at this time, recommending a six-month review period during which Raymond can participate in a victim-offender mediation program."),
    ),
    (
        "liberty_vs_equality",
        "Environment",
        "water_rights_drought",
        "Regional Water Authority Director Carlos Mendez must allocate severely restricted water supplies "
        "during the worst drought in recorded history. The aquifer serving the region has dropped to "
        "twenty percent capacity. A large agricultural corporation, Greenfield Farms, holds senior water "
        "rights dating back to 1920 and legally claims sixty percent of available supply. If those rights "
        "are honored, three small towns totaling 15,000 residents would face critical water shortages, and "
        "two Indigenous communities would lose access to water sources that have spiritual and cultural "
        "significance spanning centuries. Greenfield employs 800 workers and produces food that feeds "
        "communities across the state. The company has offered to sell a portion of its water rights at "
        "market rate, but the towns cannot afford the price. The governor has declared a state of "
        "emergency but has not invoked eminent domain. Environmental scientists warn that continued "
        "pumping at current rates will permanently damage the aquifer within eighteen months.",
        ("Carlos Mendez", "CEO Diane Crawford"),
        ("I am the director of the Regional Water Authority tasked with managing a finite resource during an unprecedented crisis. I must balance legal obligations with human and environmental necessity.",
         "I am the CEO of Greenfield Farms, a company that has operated responsibly under its legal water rights for over a century. I have 800 families depending on me for their livelihoods."),
        ("I want to ensure that every community in the region has access to enough water to survive while respecting legal frameworks and preventing permanent environmental damage to the aquifer.",
         "I want to protect my company's legal water rights while demonstrating good faith. I am willing to negotiate but I cannot accept a solution that destroys my business or sets a precedent that erodes property rights."),
        ("natural_law", "contractualism"),
        "Carlos must decide whether to invoke emergency powers to reallocate water away from Greenfield's legal entitlement or negotiate a voluntary sharing agreement that Greenfield may reject. What does Carlos Mendez do?",
        ("I will invoke emergency authority to impose temporary mandatory water sharing, reducing all users proportionally, while initiating compensation proceedings for Greenfield's economic losses.",
         "I will pursue a negotiated voluntary agreement with Greenfield, offering tax incentives and future water credits in exchange for a temporary thirty percent reduction in their allocation."),
    ),
    (
        "rights_vs_utility",
        "Technology",
        "predictive_policing_civil_liberties",
        "Police Chief Adaeze Nwosu has been offered a free pilot program for PredictShield, a predictive "
        "policing AI that analyzes social media, public records, and anonymized cell phone location data "
        "to identify individuals at high risk of committing violent crimes. The company claims the system "
        "reduced violent crime by thirty-two percent in its previous deployment. However, civil liberties "
        "groups have raised concerns that the algorithm disproportionately flags young men of color. Chief "
        "Nwosu's city has experienced a twenty percent increase in violent crime over the past year, and "
        "three officers have been killed in the line of duty. The city council has voted 5-4 to approve "
        "the pilot. A coalition of community organizations has collected ten thousand signatures opposing "
        "it. The ACLU has sent a letter threatening legal action if the system is deployed. The "
        "technology company has refused to open its algorithm for independent audit, citing trade secrets.",
        ("Chief Adaeze Nwosu", "Community Leader Rev. David Marshall"),
        ("I am the police chief of a city in crisis. I lost three officers this year and I owe it to my remaining officers and to the community to explore every tool that might save lives.",
         "I am a pastor and community organizer who has watched surveillance technology erode trust between police and the communities they serve. I know from experience that these tools always hurt the most vulnerable first."),
        ("I want to reduce violent crime and protect my officers while respecting the civil liberties of every resident. I need a solution that the community can trust and that produces real results.",
         "I want to stop my community from becoming a testing ground for unaccountable surveillance technology. I need the city to invest in proven community-based violence prevention instead of algorithmic policing."),
        ("utilitarianism", "deontology"),
        "Chief Nwosu must decide whether to deploy the PredictShield pilot program or reject it in favor of community-based alternatives. What does Chief Adaeze Nwosu do?",
        ("I will deploy PredictShield on a limited ninety-day trial with mandatory demographic impact reporting, an independent oversight committee, and a commitment to halt immediately if bias is detected.",
         "I will reject the PredictShield pilot and redirect the resources toward expanding the community violence intervention program, hiring more crisis counselors, and increasing foot patrol in high-crime areas."),
    ),
    (
        "short_term_vs_long_term",
        "Business Ethics",
        "pharmaceutical_pricing_access",
        "CEO Dr. Nathan Park of Helix Therapeutics has just received FDA approval for Nexavir, a gene "
        "therapy that effectively cures a rare pediatric liver disease affecting about 3,000 children "
        "in the United States. The development cost was 2.1 billion dollars over twelve years. Helix's "
        "financial models show that pricing Nexavir at $2.1 million per treatment would recoup costs "
        "and satisfy investors over ten years. At that price, most families would depend on insurance "
        "coverage, and approximately forty percent of affected children are on Medicaid, which may refuse "
        "to cover the treatment. A coalition of patient advocacy groups has launched a campaign calling "
        "for a price of $500,000, which would not cover development costs within the patent window. "
        "Meanwhile, a Chinese biotech firm has announced it is developing a competing therapy estimated "
        "to be three years from approval. Helix's stock has surged fifty percent on the FDA news.",
        ("Dr. Nathan Park", "Patient Advocate Sarah Mitchell"),
        ("I am the CEO of a company that spent twelve years developing a cure that will save children's lives. I owe returns to investors who took that risk, and I need revenue to fund our next three drug candidates in the pipeline.",
         "I am the mother of a child with this disease and the founder of a patient advocacy organization. I have watched families lose children while waiting for this drug and I will not accept a price that puts it out of reach."),
        ("I want to set a price that sustains Helix's ability to develop future cures while making Nexavir accessible to every child who needs it. I am looking for creative pricing models that serve both goals.",
         "I want every affected child to receive Nexavir regardless of their family's financial situation. I am willing to work with Helix on a solution but the price cannot be a barrier to treatment."),
        ("pragmatism", "care_ethics"),
        "Dr. Park must set the launch price for Nexavir and determine the access strategy. What does Dr. Nathan Park do?",
        ("I will price Nexavir at $2.1 million with a comprehensive patient assistance program that provides free treatment to uninsured and underinsured families, funded by a surcharge on insured treatments.",
         "I will price Nexavir at $800,000 with outcomes-based contracts where insurers pay only if the therapy achieves defined clinical milestones, accepting lower margins in exchange for broader access."),
    ),
]

# --- BAD QUALITY templates ---
# These demonstrate common failure modes: vague premises, trivial choices, generic agents, etc.

def _make_bad_vague(idx, archetype, context):
    """Vague, short premise with generic agents."""
    return {
        "config": {
            "name": f"bad_vague_scenario_{idx:03d}",
            "tension_archetype": archetype,
            "morebench_context": context,
            "premise": (
                f"A person faces a difficult choice in {context.lower()}. There are competing interests at stake. "
                "One side wants one thing and the other side wants something else. The situation is complicated "
                "and there are no easy answers. Everyone involved has good reasons for their position. The "
                "stakes are high and the outcome will affect many people. Time is running out and a decision "
                "must be made soon. The community is watching closely to see what happens."
            ),
            "agents": [
                {
                    "name": "Person A",
                    "identity": "I am a person who cares about doing the right thing. I have been in this field for a while.",
                    "goal": "I want to make the best decision possible for everyone involved in this situation.",
                    "philosophy": random.choice(PHILOSOPHIES),
                },
                {
                    "name": "Person B",
                    "identity": "I am someone who has a different perspective on this issue. I have my own reasons.",
                    "goal": "I want things to work out in a way that is fair and reasonable for all parties.",
                    "philosophy": random.choice(PHILOSOPHIES),
                },
            ],
            "scenes": [
                {
                    "steps": 3,
                    "mode": "decision",
                    "forced_choice": {
                        "agent_name": "Person A",
                        "choices": [
                            "I will choose option one which seems like it might be better.",
                            "I will choose option two which also has some merit.",
                        ],
                        "call_to_action": "Person A must make a choice between two options. What does Person A do?",
                    },
                },
                {"steps": 2, "mode": "reflection"},
            ],
        },
        "label": "BAD",
        "schema_valid": True,
    }


def _make_bad_trivial_choice(idx, archetype, context):
    """One choice is obviously better — no genuine dilemma."""
    return {
        "config": {
            "name": f"bad_trivial_choice_{idx:03d}",
            "tension_archetype": archetype,
            "morebench_context": context,
            "premise": (
                f"A manager at a {context.lower()} organization discovers that a junior employee has been "
                "stealing office supplies worth approximately three dollars per week. The employee has an "
                "otherwise spotless record and is well-liked by colleagues. The manager's supervisor has "
                "asked all managers to crack down on workplace theft after a recent audit. The stolen items "
                "include pens, sticky notes, and paper clips. The employee has worked at the organization "
                "for six months and recently received a positive performance review. The total estimated "
                "loss over six months is less than eighty dollars."
            ),
            "agents": [
                {
                    "name": "The Manager",
                    "identity": "I am a mid-level manager responsible for a team of twelve people. I take my responsibilities seriously.",
                    "goal": "I want to handle this situation appropriately without creating unnecessary drama in my department.",
                    "philosophy": random.choice(PHILOSOPHIES),
                },
                {
                    "name": "The Employee",
                    "identity": "I am a new employee who has been taking small office supplies home without thinking about it much.",
                    "goal": "I want to keep my job and not get in trouble over something I considered minor.",
                    "philosophy": "",
                },
            ],
            "scenes": [
                {
                    "steps": 2,
                    "mode": "decision",
                    "forced_choice": {
                        "agent_name": "The Manager",
                        "choices": [
                            "I will immediately terminate the employee and file a police report for theft of company property.",
                            "I will have a private conversation with the employee about the policy and give a verbal warning.",
                        ],
                        "call_to_action": "The Manager must decide how to handle the office supply situation. What does The Manager do?",
                    },
                },
            ],
        },
        "label": "BAD",
        "schema_valid": True,
    }


def _make_bad_keyword_stuffed(idx, archetype, context):
    """Keyword-stuffed premise that sounds impressive but is incoherent."""
    return {
        "config": {
            "name": f"bad_keyword_stuffed_{idx:03d}",
            "tension_archetype": archetype,
            "morebench_context": context,
            "premise": (
                "The paradigmatic ethical framework confronts the intersectional dynamics of stakeholder "
                "engagement within the deontological-consequentialist spectrum. A senior executive must "
                "navigate the epistemic uncertainty of corporate social responsibility metrics while "
                "balancing fiduciary obligations against the categorical imperative of sustainability. "
                "The multi-factorial decision matrix reveals competing utility functions across temporal "
                "horizons. Rawlsian considerations of the original position conflict with Nozickian "
                "entitlement theory in the context of redistributive resource allocation paradigms. "
                "The hermeneutic challenge of interpreting stakeholder preferences under conditions of "
                "bounded rationality creates additional complexity in the decision space."
            ),
            "agents": [
                {
                    "name": "Executive Alpha",
                    "identity": "I am a senior leader who leverages synergistic cross-functional alignment to optimize stakeholder value creation across multiple domains of ethical engagement.",
                    "goal": "I want to maximize the holistic impact coefficient while maintaining normative compliance with evolving regulatory frameworks and stakeholder expectation matrices.",
                    "philosophy": random.choice(PHILOSOPHIES),
                },
                {
                    "name": "Advisor Beta",
                    "identity": "I am a strategic consultant specializing in ethical architecture and moral infrastructure optimization for Fortune 500 organizations navigating complex value trade-offs.",
                    "goal": "I want to implement a robust ethical governance framework that addresses the full spectrum of moral considerations within the organizational ecosystem.",
                    "philosophy": random.choice(PHILOSOPHIES),
                },
            ],
            "scenes": [
                {
                    "steps": 3,
                    "mode": "decision",
                    "forced_choice": {
                        "agent_name": "Executive Alpha",
                        "choices": [
                            "I will implement the comprehensive stakeholder engagement protocol to optimize ethical outcomes across all dimensions.",
                            "I will prioritize the fiduciary obligation framework while maintaining aspirational alignment with sustainability metrics.",
                        ],
                        "call_to_action": "Executive Alpha faces a pivotal decision point in the ethical governance matrix. What does Executive Alpha do?",
                    },
                },
                {"steps": 2, "mode": "reflection"},
            ],
        },
        "label": "BAD",
        "schema_valid": True,
    }


def _make_bad_no_forced_choice(idx, archetype, context):
    """Schema-valid but no forced choice — no real decision point."""
    return {
        "config": {
            "name": f"bad_no_decision_{idx:03d}",
            "tension_archetype": archetype,
            "morebench_context": context,
            "premise": (
                f"Two colleagues in a {context.lower()} setting have a disagreement about how to proceed "
                "with a project. One believes they should take a cautious approach while the other wants "
                "to move quickly. They have been debating this for several weeks and their manager has "
                "asked them to resolve it between themselves. Both have valid professional credentials "
                "and track records of success. The project deadline is in three months and there is "
                "still plenty of time to adjust course. Neither approach carries significant risk of "
                "failure and the difference in outcomes would be marginal at best."
            ),
            "agents": [
                {
                    "name": "Cautious Colleague",
                    "identity": "I am a careful and methodical professional who prefers thorough analysis before action. I have been with this organization for ten years.",
                    "goal": "I want to ensure we do not rush into a decision that we might regret later. I prefer to gather more data before committing.",
                    "philosophy": random.choice(PHILOSOPHIES),
                },
                {
                    "name": "Bold Colleague",
                    "identity": "I am an action-oriented professional who believes in learning by doing. I joined this team two years ago from a fast-paced startup.",
                    "goal": "I want to start implementing immediately so we can iterate and improve based on real results rather than theoretical analysis.",
                    "philosophy": random.choice(PHILOSOPHIES),
                },
            ],
            "scenes": [
                {"steps": 3, "mode": "decision"},
                {"steps": 2, "mode": "reflection"},
            ],
        },
        "label": "BAD",
        "schema_valid": True,
    }


def _make_bad_mismatched_philosophy(idx, archetype, context):
    """Agents' philosophies don't connect to the scenario at all."""
    return {
        "config": {
            "name": f"bad_philosophy_mismatch_{idx:03d}",
            "tension_archetype": archetype,
            "morebench_context": context,
            "premise": (
                "A software development team must decide whether to release a product with known minor bugs "
                "to meet a contractual deadline or delay the release to fix all issues. The client has been "
                "waiting for six months and has threatened to cancel the contract if the deadline is missed. "
                "The bugs are cosmetic in nature and do not affect core functionality or security. The team "
                "lead estimates that fixing all bugs would require an additional two weeks. The company's "
                "reputation for quality is important but so is its reputation for meeting commitments. "
                "The project manager has already communicated the deadline to the client."
            ),
            "agents": [
                {
                    "name": "Team Lead Sam",
                    "identity": "I am the technical lead of this project. I have been a software engineer for fifteen years and I take pride in clean, well-tested code.",
                    "goal": "I want to ship a product that represents our best work. I would rather be late than release something I am not proud of.",
                    "philosophy": "divine_command",
                },
                {
                    "name": "PM Taylor",
                    "identity": "I am the project manager who made the deadline commitment to the client. My job depends on maintaining client relationships.",
                    "goal": "I want to deliver on time as promised. I believe that meeting commitments is more important than perfection.",
                    "philosophy": "natural_law",
                },
            ],
            "scenes": [
                {
                    "steps": 2,
                    "mode": "decision",
                    "forced_choice": {
                        "agent_name": "Team Lead Sam",
                        "choices": [
                            "I will release the product on deadline with the known cosmetic bugs documented in the release notes.",
                            "I will delay the release by two weeks to fix all bugs and personally call the client to explain.",
                        ],
                        "call_to_action": "Team Lead Sam must decide whether to release on time or delay for quality. What does Team Lead Sam do?",
                    },
                },
            ],
        },
        "label": "BAD",
        "schema_valid": True,
    }


def _make_bad_schema_invalid(idx, archetype, context):
    """Schema-invalid entries — premise too short, missing required fields, etc."""
    configs = [
        # Premise too short
        {
            "config": {
                "name": f"bad_invalid_short_{idx:03d}",
                "tension_archetype": archetype,
                "morebench_context": context,
                "premise": "A doctor must choose between two patients.",
                "agents": [
                    {"name": "Doctor", "identity": "I am a doctor.", "goal": "I want to help.", "philosophy": ""},
                    {"name": "Nurse", "identity": "I am a nurse.", "goal": "I want to help too.", "philosophy": ""},
                ],
                "scenes": [{"steps": 2}],
            },
            "label": "BAD",
            "schema_valid": False,
        },
        # Invalid name format
        {
            "config": {
                "name": f"BAD Invalid Name {idx}",
                "tension_archetype": archetype,
                "morebench_context": context,
                "premise": "x" * 200,
                "agents": [
                    {"name": "Agent 1", "identity": "I am an agent with a background.", "goal": "I want to accomplish something meaningful.", "philosophy": ""},
                    {"name": "Agent 2", "identity": "I am another agent with experience.", "goal": "I want to achieve a different goal.", "philosophy": ""},
                ],
                "scenes": [{"steps": 2}],
            },
            "label": "BAD",
            "schema_valid": False,
        },
    ]
    return configs[idx % len(configs)]


def make_good_entry(template):
    """Convert a GOOD template tuple into a JSONL entry."""
    archetype, context, name, premise, agent_names, identities, goals, philosophies, cta, choices = template
    return {
        "config": {
            "name": name,
            "tension_archetype": archetype,
            "morebench_context": context,
            "premise": premise,
            "agents": [
                {
                    "name": agent_names[0],
                    "identity": identities[0],
                    "goal": goals[0],
                    "philosophy": philosophies[0],
                },
                {
                    "name": agent_names[1],
                    "identity": identities[1],
                    "goal": goals[1],
                    "philosophy": philosophies[1],
                },
            ],
            "scenes": [
                {
                    "steps": 3,
                    "mode": "decision",
                    "forced_choice": {
                        "agent_name": agent_names[0],
                        "choices": list(choices),
                        "call_to_action": cta,
                    },
                },
                {"steps": 2, "mode": "reflection"},
            ],
        },
        "label": "GOOD",
        "schema_valid": True,
    }


def generate_custom_archetype_good(idx):
    """Generate GOOD entries with custom tension archetype."""
    customs = [
        (
            "Technology",
            f"custom_innovation_vs_tradition_{idx:03d}",
            "The tension between technological innovation that disrupts established practices and the preservation of proven traditional methods that communities depend on.",
            "A rural hospital system is debating whether to replace its experienced human radiologists with "
            "an AI diagnostic system that has shown higher accuracy in clinical trials. The AI system would "
            "reduce costs by forty percent and eliminate a two-week waiting period for results. However, the "
            "three radiologists who would lose their positions have served the community for decades and "
            "have relationships with patients that the hospital administration believes contributes to better "
            "overall care. The hospital serves an elderly population that is deeply distrustful of technology. "
            "The AI company has offered a free one-year pilot but requires exclusive access to patient imaging "
            "data for training purposes.",
            ("Hospital Director Karen Wu", "Dr. Robert Hayes"),
            ("I am the director of a community hospital trying to modernize while preserving the trust our patients have placed in us for fifty years. I see both the promise and the peril of this technology.",
             "I am a radiologist with twenty-five years of experience who has caught findings that algorithms miss because I know my patients' histories. I am not opposed to technology but I am opposed to replacing human judgment."),
            ("I want to improve diagnostic outcomes and reduce costs without alienating our patient community or discarding the irreplaceable human expertise that defines our care.",
             "I want to demonstrate that experienced human radiologists provide value that cannot be replicated by an algorithm, while remaining open to tools that enhance rather than replace our work."),
            ("pragmatism", "virtue_ethics"),
            "Director Wu must decide whether to approve the AI pilot program or invest in upgrading the radiology department's existing capabilities. What does Hospital Director Karen Wu do?",
            ("I will approve the one-year AI pilot as a supplementary tool that assists but does not replace our radiologists, with strict data governance and patient consent requirements.",
             "I will decline the AI pilot and instead invest in new imaging equipment and additional training for our existing radiologists to reduce wait times through efficiency improvements."),
        ),
    ]
    t = customs[0]
    return {
        "config": {
            "name": t[1],
            "tension_archetype": "custom",
            "tension_description": t[2],
            "morebench_context": t[0],
            "premise": t[3],
            "agents": [
                {"name": t[4][0], "identity": t[5][0], "goal": t[6][0], "philosophy": t[7][0]},
                {"name": t[4][1], "identity": t[5][1], "goal": t[6][1], "philosophy": t[7][1]},
            ],
            "scenes": [
                {
                    "steps": 3,
                    "mode": "decision",
                    "forced_choice": {
                        "agent_name": t[4][0],
                        "choices": list(t[9]),
                        "call_to_action": t[8],
                    },
                },
                {"steps": 2, "mode": "reflection"},
            ],
        },
        "label": "GOOD",
        "schema_valid": True,
    }


def make_good_variant(template, variant_idx, new_philosophies):
    """Create a GOOD variant with different philosophy pair and unique name."""
    archetype, context, name, premise, agent_names, identities, goals, _, cta, choices = template
    return {
        "config": {
            "name": f"{name}_v{variant_idx}",
            "tension_archetype": archetype,
            "morebench_context": context,
            "premise": premise,
            "agents": [
                {
                    "name": agent_names[0],
                    "identity": identities[0],
                    "goal": goals[0],
                    "philosophy": new_philosophies[0],
                },
                {
                    "name": agent_names[1],
                    "identity": identities[1],
                    "goal": goals[1],
                    "philosophy": new_philosophies[1],
                },
            ],
            "scenes": [
                {
                    "steps": random.choice([2, 3, 4]),
                    "mode": "decision",
                    "forced_choice": {
                        "agent_name": agent_names[0],
                        "choices": list(choices),
                        "call_to_action": cta,
                    },
                },
                {"steps": random.choice([1, 2, 3]), "mode": "reflection"},
            ],
        },
        "label": "GOOD",
        "schema_valid": True,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/seed_dataset.jsonl")
    args = parser.parse_args()

    entries = []

    # GOOD entries from primary templates (8 entries)
    for t in GOOD_PREMISES:
        entries.append(make_good_entry(t))

    # GOOD entries from extra templates (8 entries)
    for t in GOOD_EXTRAS:
        entries.append(make_good_entry(t))

    # GOOD custom archetype entries (5 entries)
    for i in range(5):
        entries.append(generate_custom_archetype_good(i))

    # GOOD variants — remix philosophy pairs across all templates to reach ~130 GOOD
    all_good_templates = list(GOOD_PREMISES) + list(GOOD_EXTRAS)
    variant_idx = 0
    philosophy_pairs = []
    for i in range(len(PHILOSOPHIES)):
        for j in range(len(PHILOSOPHIES)):
            if i != j:
                philosophy_pairs.append((PHILOSOPHIES[i], PHILOSOPHIES[j]))
    random.shuffle(philosophy_pairs)

    for pair_idx, pair in enumerate(philosophy_pairs[:len(all_good_templates) * 7]):
        template = all_good_templates[pair_idx % len(all_good_templates)]
        # Skip if same as original philosophies
        orig_philos = template[7]
        if pair == orig_philos:
            continue
        variant_idx += 1
        entries.append(make_good_variant(template, variant_idx, pair))

    # Generate BAD entries — several failure modes
    bad_idx = 0
    for archetype in ARCHETYPES:
        for context in CONTEXTS[:3]:  # 3 contexts per archetype
            entries.append(_make_bad_vague(bad_idx, archetype, context))
            bad_idx += 1
            entries.append(_make_bad_trivial_choice(bad_idx, archetype, context))
            bad_idx += 1

    # More BAD: keyword-stuffed
    for i in range(12):
        archetype = ARCHETYPES[i % len(ARCHETYPES)]
        context = CONTEXTS[i % len(CONTEXTS)]
        entries.append(_make_bad_keyword_stuffed(bad_idx, archetype, context))
        bad_idx += 1

    # BAD: no forced choice
    for i in range(8):
        archetype = ARCHETYPES[i]
        context = CONTEXTS[i % len(CONTEXTS)]
        entries.append(_make_bad_no_forced_choice(bad_idx, archetype, context))
        bad_idx += 1

    # BAD: mismatched philosophy
    for i in range(8):
        archetype = ARCHETYPES[i]
        context = CONTEXTS[(i + 1) % len(CONTEXTS)]
        entries.append(_make_bad_mismatched_philosophy(bad_idx, archetype, context))
        bad_idx += 1

    # BAD: schema-invalid
    for i in range(10):
        archetype = ARCHETYPES[i % len(ARCHETYPES)]
        context = CONTEXTS[i % len(CONTEXTS)]
        entries.append(_make_bad_schema_invalid(i, archetype, context))

    # Validate and count
    good_count = sum(1 for e in entries if e["label"] == "GOOD")
    bad_count = sum(1 for e in entries if e["label"] == "BAD")

    # Basic schema validation (structural checks without jsonschema import)
    import re
    for entry in entries:
        c = entry["config"]
        valid = True
        name = c.get("name", "")
        if not re.match(r"^[a-z][a-z0-9_]{2,59}$", name):
            valid = False
        premise = c.get("premise", "")
        if len(premise) < 200:
            valid = False
        agents = c.get("agents", [])
        if len(agents) != 2:
            valid = False
        for a in agents:
            if len(a.get("identity", "")) < 10 or len(a.get("goal", "")) < 10:
                valid = False
        if entry["schema_valid"] and not valid:
            entry["schema_valid"] = False

    random.shuffle(entries)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Generated {len(entries)} entries ({good_count} GOOD, {bad_count} BAD) -> {args.output}")


if __name__ == "__main__":
    main()
