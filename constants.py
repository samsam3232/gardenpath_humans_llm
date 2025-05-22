import random
from copy import deepcopy
from typing import List


ZS_PROMPT_YN = "SENTENCE\n\nAnswer with Yes or No:\nQUESTION\n\n"

YN_1 = """The doctor that the nurse called checked on the patient yesterday.

Answer with Yes or No:
Did the nurse call the doctor?
Yes"""


YN_2 = """The teacher that helped the student graded the papers on the weekend.

Answer with Yes or No:
Did the student grade the papers?
No"""


YN_3 = """The sailor that the captain punished stayed in his room.

Answer with Yes or No:
Did the captain stay in his room?
No"""


YN_4 = """The driver that saved the cyclist went back home.

Answer with Yes or No:
Did the driver go back home?
Yes"""


YN_5 = """The dentist that the children feared was, in reality, really gentle.

Answer with Yes or No:
Was the dentist really gentle?
Yes"""


YN_6 = """The singer that hired the guitarist arrived to the concert early.

Answer with Yes or No:
Did the singer hire the guitarist?
Yes"""


YN_7 = """The professor that emailed the surgeon was stuck on a case.

Answer with Yes or No:
Did the surgeon email the professor?
No"""


YN_8 = """The dog that the rescuers found stayed in the mountains for a week.

Answer with Yes or No:
Did the dog find the rescuers?
No"""

EXAMPLES_YN = [YN_1, YN_2, YN_3, YN_4, YN_5, YN_6, YN_7, YN_8]


ZS_PROMPT_YN_COT = "SENTENCE\n\nAnswer with Yes or No. Explain your answer first:\nQUESTION\n\n"

YN_1_COT = """The doctor that the nurse called checked on the patient yesterday.

Answer with Yes or No. Explain your answer first:
Did the nurse call the doctor?

Explanation: In the sentence, we first see "the doctor that the called". This means that the nurse called the doctor. Therefore, the answer is Yes.

Answer: Yes"""


YN_2_COT = """The teacher that helped the student graded the papers on the weekend.

Answer with Yes or No. Explain your answer first:
Did the student grade the papers?

Explanation: In the sentence, we first see "the teacher that helped the student". This means that the teacher helped the student. Then we see "graded the papers". This means that the teacher graded the papers. Therefore, the answer is No.

Answer: No"""


YN_3_COT = """The sailor that the captain punished stayed in his room.

Answer with Yes or No. Explain your answer first:
Did the captain stay in his room?

Explanation: In the sentence, we first see "the sailor that the captain punished". This means that the captain punished the sailor. Then we see "stayed in his room". This means that the sailor stayed in his room, not the captain. Therefore, the answer is No.

Answer: No"""


YN_4_COT = """The driver that saved the cyclist went back home.

Answer with Yes or No. Explain your answer first:
Did the driver go back home?

Explanation: In the sentence, we first see "the driver that saved the cyclist". This means that the driver saved the cyclist. Then we see "went back home". This means that the driver went back home. Therefore, the answer is Yes.

Answer: Yes"""


YN_5_COT = """The dentist that the children feared was, in reality, really gentle.

Answer with Yes or No. Explain your answer first:
Was the dentist really gentle?

Explanation: In the sentence, we first see "the dentist that the children feared". This means that the children feared the dentist. Then we see "really gentle". This means that the dentist was really gentle. Therefore, the answer is Yes.

Answer: Yes"""


YN_6_COT = """The singer that hired the guitarist arrived to the concert early.

Answer with Yes or No. Explain your answer first:
Did the singer hire the guitarist?

Explanation: In the sentence, we first see "the singer that hired the guitarist". This means that the singer hired the guitarist. Therefore, the answer is Yes.

Answer: Yes"""


YN_7_COT = """The professor that emailed the surgeon was stuck on a case.

Answer with Yes or No. Explain your answer first:
Did the surgeon email the professor?

Explanation: In the sentence, we first see "the professor that emailed the surgeon". This means that the professor emailed the surgeon. Therefore, the answer is No.

Answer: No"""


YN_8_COT = """The dog that the rescuers found stayed in the mountains for a week.

Answer with Yes or No. Explain your answer first:
Did the dog find the rescuers?

Explanation: In the sentence, we first see "the dog that the rescuers found". This means that the rescuers found the dog. Therefore, the answer is No.

Answer: No"""

EXAMPLES_YN_COT = [YN_1_COT, YN_2_COT, YN_3_COT, YN_4_COT, YN_5_COT, YN_6_COT, YN_7_COT, YN_8_COT]

PROMPTS_MAPPING = {"yes_no": EXAMPLES_YN, "yes_no_cot": EXAMPLES_YN_COT}

ZS_MAPPING = {"yes_no": ZS_PROMPT_YN, "yes_no_cot": ZS_PROMPT_YN_COT}


DEFAULT_SYSTEM = """You are a linguistic experiment subject. You will be presented with a sentence, and will need to answer a \
reading comprehension question. You will need to select an option amongst the proposed answers.
Here are a few examples of questions and relevant answers:

EXAMPLES"""

DEFAULT_SYSTEM_2 = """You will answer a reading comprehension question about a sentence.
Here are a few examples of questions and correct answers:

EXAMPLES"""


DEFAULT_QUESTION = """Here is the sentence:
SENTENCE

Answer this question:
QUESTION"""

PREFIX = """You are a linguistic experiment subject. You will be presented with a sentence, and will need to answer a \
reading comprehension question. You will need to select an option amongst the proposed answers.
Here are a few examples of questions and relevant answers:

EXAMPLES

Here is the sentence:
SENTENCE

Answer this question:
QUESTION
"""

OPTIONS = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M",
           14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S"}


def get_examples(question_type: str = "yes_no", num_examples: int = 4, num_samples: int = 8) -> List:

    """
    Returns shuffled lists of examples for a question type
    """

    sampled_examples = list()
    for _ in range(num_samples):
        new_examples = deepcopy(PROMPTS_MAPPING[question_type])
        random.shuffle(new_examples)
        sampled_examples.append("\n\n\n".join(new_examples[:num_examples]))

    return sampled_examples