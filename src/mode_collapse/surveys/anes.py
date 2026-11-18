"""ANES questions, demographic groups, and prompts for Experiment 2."""

from __future__ import annotations

import pandas as pd

from mode_collapse.surveys.gss import (
    ANSWER_LETTERS,
    SEED,
    build_chat_messages,
    build_question_block,
)

DATASET = "anes"
CELL_VARS = ["decade", "age", "sex", "race", "region", "education"]
ITEM_QUESTIONS = {
    "VCF0301": (
        "Generally speaking, do you usually think of yourself as a Republican, a Democr"
        "at, an Independent, or what? (IF REPUBLICAN OR DEMOCRAT) Would you call yourse"
        "lf a strong (REP/DEM) or a not very strong (REP/DEM)? (IF INDEPENDENT, OTHER ["
        "1966 AND LATER:] OR NO PREFERENCE; 2008: OR DK) Do you think of yourself as cl"
        "oser to the Republican or Democratic party?"
    ),
    "VCF0302": (
        "Generally speaking, do you usually think of yourself as a Republican, a Democr"
        "at, an Independent, or what?"
    ),
    "VCF0305": "Party ID - strength of partisanship",
    "VCF0310": (
        "Some people don’t pay much attention to political campaigns. How about you, wo"
        "uld you say that you have been/were very much interested, somewhat interested,"
        " or not much interested in (1952-1998: following) the political campaigns (so "
        "far) this year?"
    ),
    "VCF0343": "President trait - moral",
    "VCF0450": (
        "Do you approve or disapprove of the way that [the president] is handling his/h"
        "er job as President?"
    ),
    "VCF0451": (
        "Do you approve or disapprove of the way that [the president] is handling his/h"
        "er job as President? Do you (approve/disapprove) {of the president’s performan"
        "ce} strongly or not strongly?"
    ),
    "VCF0501": (
        "Do you think there are any important diﬀerences in what the Republicans and De"
        "mocrats stand for?"
    ),
    "VCF0502": (
        "Would you say that one of the parties (1972: either party) is more conservativ"
        "e or more liberal than the other (1970-1976: at the national level)? (IF YES:)"
        " Which party is more conservative? (IF NO 1960,1968:) Do you think that people"
        " generally consider the Democrats or the Republicans more conservative or woul"
        "dn’t you want to guess about that? (IF NO OR DK 1964,1970,1972,1976:) Do you t"
        "hink that "
    ),
    "VCF0503": (
        "We hear a lot of talk these days about liberals and conservatives. I’m going t"
        "o show you (1996 AND LATER: Here is) a seven-point scale on which the politica"
        "l views that people might hold are arranged 261 from extremely liberal to extr"
        "emely conservative. Where would you place the Democratic Party {on this scale}"
        "? (7-POINT SCALE SHOWN TO R)"
    ),
    "VCF0504": "Republican Party - R’s rating of party on liberal/conservative scale",
    "VCF0605": (
        "Would you say the government is pretty much run by a few big interests looking"
        " out for themselves or that it is run for the benefit of all the people?"
    ),
    "VCF0606": (
        "Do you think that people in the government waste a lot of money we pay in taxe"
        "s, waste some of it, or don’t waste very much of it?"
    ),
    "VCF0609": (
        "Now I’d like to read some of the kinds of things people tell us when we interv"
        "iew them. Please tell me whether (1992: and ask you whether) (1966,1988,1990,1"
        "994-LATER: how much) you agree or disagree with these statements (1992: with t"
        "hem) (2002: about the government.) (1988,1990: You can just give me the number"
        " of your choice.) (1992: I’ll read them one at a time and you just tell me whe"
        "ther you a"
    ),
    "VCF0613": "‘People like me don’t have any say about what the government does.’",
    "VCF0624": (
        "(And) How much do you feel that having elections makes the government pay atte"
        "ntion to what the people think, a good deal, some or not much?"
    ),
    "VCF0803": (
        "We hear a lot of talk these days about liberals and conservatives. Here is (19"
        "72,1974: I’m going to show you) a 7-point scale on which the political views t"
        "hat people might hold are arranged from extremely liberal to extremely conserv"
        "ative. Where would you place yourself on this scale, or haven’t you thought mu"
        "ch about this? (7-POINT SCALE SHOWN TO R) When it comes to politics, do you us"
        "ually thin"
    ),
    "VCF0806": (
        "There is much concern about the rapid rise in medical and hospital costs. Some"
        " (1988,1994-LATER: people) feel there should be a government insurance plan wh"
        "ich would cover all medical and hospital expenses (1984 AND LATER: for everyon"
        "e). (1996,2004: Suppose these people are at one end of a scale, at point 1). O"
        "thers feel that (1988,1994-1996: all) medical expenses should be paid by indiv"
        "iduals, an"
    ),
    "VCF0819": (
        "Which of these statements would you agree with: (STATEMENTS CONTAINED IN CODE "
        "CATEGORIES READ TO R) Some people say that Negroes should be allowed to live i"
        "n any part of town they want to. How do you feel? Should Negroes be allowed to"
        " live in any part of town they want to, or not?"
    ),
    "VCF0823": (
        "(Same introduction as in VCF0805 [CARD WITH RESPONSES SHOWN]). Now I’d like to"
        " read some of the things people tell us when we interview them (1968: and ask "
        "you; 1980: As I read, please tell me) whether you agree or disagree with them."
        " 1972: I’d like you to tell me whether you agree or disagree with each of thes"
        "e next six statements. 1976: I am going to read you two statements about US fo"
        "reign poli"
    ),
    "VCF0825": (
        "How about the chances of our country getting into a bigger war? Compared to a "
        "few years ago, do you think we are more likely, less likely, or have about the"
        " same chances to get into a bigger war?"
    ),
    "VCF0830": (
        "Some people feel that the government in Washington should make every possible "
        "eﬀort to improve the social and economic position of blacks (1970: Negroes) an"
        "d other minority groups (1980: even if it means giving them preferential treat"
        "ment). Others feel that the government should not make any special eﬀort to he"
        "lp minorities because they should help themselves (1970: but they should be ex"
        "pected to "
    ),
    "VCF0838": (
        "There has been some discussion about abortion during recent years. (RESPONDENT"
        " BOOKLET) Which one of the opinions on this page best agrees with your view? Y"
        "ou can just tell me the number of the opinion you choose. I am going to read y"
        "ou a short list of opinions. Please tell me which 360 VARIABLE DESCRIPTION one"
        " of the opinions best agrees with your view. You can just tell me the number o"
        "f the opin"
    ),
    "VCF0839": (
        "Some people think the government should provide fewer services, even in areas "
        "such as health and education, in order to reduce spending. (2004: Suppose thes"
        "e people are at one end of a scale, at point 1.) Other people feel that it is "
        "important for the government to provide many more services even if it means an"
        " increase in spending. (2004: Suppose these people are at the other end, at po"
        "int 7. And"
    ),
    "VCF0842": (
        "Some people think we need much tougher government regulations on business in o"
        "rder to protect the environment. (Suppose these people are at one end of a sca"
        "le, at point 1.) Other think that current regulations to protect the environme"
        "nt are already too much of a burden on business. (Suppose these people are at "
        "the other end of the scale, at point 7.) And, of course, some other people hav"
        "e opinions"
    ),
    "VCF0843": (
        "Some people believe that we should spend much less money for defense. (1996,20"
        "04: Suppose these people are at one end of a scale, at point 1.) Others feel t"
        "hat defense spending should be greatly increased. (1996,2004: Suppose these pe"
        "ople are at the other end, at point 7.) (2004: And, of course, some other peop"
        "le have opinions somewhere in between, at points 2,3,4,5, or 6). Where would y"
        "ou place y"
    ),
    "VCF0846": (
        "Now on another topic. . .) Do you consider religion to be an important part of"
        " your life, or not?"
    ),
    "VCF0850": (
        "I am going to read you a short list of statements. Please tell me) Which of th"
        "ese statements comes closest to describing your feelings about the Bible? You "
        "can just give me the number of the statement you choose. Please look at page ["
        "page] of the booklet. Which of these statements comes closest to describing yo"
        "ur feelings about the Bible? You can just give me the number of your choice. ("
        "STATEMENTS"
    ),
    "VCF0852": (
        "‘The world is always changing and we should adjust our view of moral behavior "
        "to those changes.’ (2000,2004: do you agree strongly, agree somewhat, neither "
        "agree nor disagree, disagree somewhat, or disagree strongly with this statemen"
        "t?)"
    ),
    "VCF0853": (
        "‘This country would have many fewer problems if there were more emphasis on tr"
        "aditional family ties.’ (2000,2004: do you agree strongly, agree somewhat, nei"
        "ther agree nor disagree, disagree somewhat, or disagree strongly with this sta"
        "tement?)"
    ),
    "VCF0870": (
        "How about (1996-LATER: Now thinking about) the economy (1990,1994- later: in t"
        "he country as a whole)? Would you say that over the past year the nation’s eco"
        "nomy has gotten better, stayed (all yrs. exc 1984: about) the same or gotten w"
        "orse? Would you say that over the past year the nation’s economy has gotten wo"
        "rse, stayed about the same, or gotten better?"
    ),
    "VCF0871": (
        "How about (1996-LATER: Now thinking about) the economy (1990,1994- later: in t"
        "he country as a whole)? Would you say that over the past year the nation’s eco"
        "nomy has gotten better, stayed (all yrs. exc 1984: about) the same or gotten w"
        "orse? (IF BETTER:) Would you say much better or somewhat better? (IF WORSE:) W"
        "ould you say much worse or somewhat worse?"
    ),
    "VCF0872": (
        "What about the next 12 months (1980,1982: or so)? Do you expect the (1986,1988"
        ",1992: national) economy to get better, get worse, or stay about the same? Do "
        "you expect the economy, in the country as a whole, to get better, stay about t"
        "he same, or get worse? Do you expect the economy, in the country as a whole, t"
        "o get worse, stay about the same, or get better?"
    ),
    "VCF0876": (
        "Do you favor or oppose laws to protect [homosexuals/gays and lesbians] against"
        " job discrimination?"
    ),
    "VCF0876a": (
        "Do you favor or oppose laws to protect [homosexuals/gays and lesbians] against"
        " job discrimination? Do you favor/oppose such laws strongly or not strongly?"
    ),
    "VCF0878": (
        "Do you think gay or lesbian couples, in other words, homosexual couples, shoul"
        "d be legally permitted to adopt children?"
    ),
    "VCF0880": (
        "We are interested in how people are getting along financially these days. Woul"
        "d you say that (1962,1966-1974: you [and your family]; 1976 and later : you [a"
        "nd your family living here]) are better oﬀ or worse oﬀ financially than you we"
        "re a year ago. Would you say that you (and your family) (2000 FACE-TO-FACE ONL"
        "Y: living here) are better oﬀ, worse oﬀ, or just about the same financially as"
        " you were "
    ),
    "VCF0880a": (
        "We are interested in how people are getting along financially these days. Woul"
        "d you say that you and your family living here are better oﬀ or worse oﬀ finan"
        "cially than you were a year ago. Would you say that you (and your family) (200"
        "0 FACE-TO-FACE ONLY: living here) are better oﬀ, worse oﬀ, or just about the s"
        "ame financially as you were a year ago? Is that much better/worse oﬀ or somewh"
        "at better/"
    ),
    "VCF0881": (
        "Now looking ahead and thinking about the next few years, do you expect your fi"
        "nancial situation will stay about the way it is now, get better, or get worse."
        " Now looking ahead–do you think that a year from now (1962,1966-1970: you peop"
        "le; 1972,1974: you [and your family]; 1976 AND LATER: you [and your family; 20"
        "00 TELEPHONE,2004: living here]) will be better oﬀ financially or worse oﬀ, or"
        " just abou"
    ),
    "VCF0886": "poor people 2000,2012-later: aid to the poor 2002,2004: aid to poor people",
    "VCF0888": "Dealing with crime",
    "VCF0890": "Public schools",
    "VCF0894": "Welfare programs",
    "VCF0991": (
        "In general, do you approve or disapprove of the way [1980-2004: running U.S.Ho"
        "use incumbent Representative; 2012: U.S. House incumbent Representative] has b"
        "een handling his/her job?"
    ),
}
ANES_ITEM_CODES = {
    "VCF0301": (
        "VCF0301",
        {
            1: "Strong Democrat",
            2: "Weak Democrat",
            3: "Independent - Democrat",
            4: "Independent - Independent",
            5: "Independent - Republican",
            6: "Weak Republican",
            7: "Strong Republican",
        },
    ),
    "VCF0302": (
        "VCF0302",
        {
            1: "Republican",
            2: "Independent",
            3: "No preference; none; neither",
            5: "Democrat",
        },
    ),
    "VCF0305": (
        "VCF0305",
        {
            1: "Independent",
            2: "Leaning Independent",
            3: "Weak Partisan",
            4: "Strong Partisan",
        },
    ),
    "VCF0310": (
        "VCF0310",
        {1: "Not much interested", 2: "Somewhat interested", 3: "Very much interested"},
    ),
    "VCF0343": (
        "VCF0343",
        {1: "Extremely well", 2: "Quite well", 3: "Not too well", 4: "Not well at all"},
    ),
    "VCF0450": ("VCF0450", {1: "Approve", 2: "Disapprove"}),
    "VCF0451": (
        "VCF0451",
        {
            1: "Approve, strongly",
            2: "Approve, not strongly",
            3: "Disapprove, not strongly",
            4: "Disapprove, strongly",
        },
    ),
    "VCF0501": ("VCF0501", {1: "No diﬀerence", 2: "Yes, a diﬀerence"}),
    "VCF0502": (
        "VCF0502",
        {
            1: "Yes, Democrats more conservative",
            2: "Yes, Republicans more conservative",
        },
    ),
    "VCF0503": (
        "VCF0503",
        {
            1: "Extremely liberal",
            2: "Liberal",
            3: "Slightly liberal",
            4: "Moderate, middle of the road",
            5: "Slightly conservative",
            6: "Conservative",
            7: "Extremely conservative",
        },
    ),
    "VCF0504": (
        "VCF0504",
        {
            1: "Extremely liberal",
            2: "Liberal",
            3: "Slightly liberal",
            4: "Moderate, middle of the road",
            5: "Slightly conservative",
            6: "Conservative",
            7: "Extremely conservative",
        },
    ),
    "VCF0605": ("VCF0605", {1: "Few big interests", 2: "Benefit of all"}),
    "VCF0606": ("VCF0606", {1: "A lot", 2: "Some", 3: "Not very much"}),
    "VCF0609": (
        "VCF0609",
        {
            1: "Agree",
            2: "Disagree",
            3: "Neither agree nor disagree (1988 and later only)",
        },
    ),
    "VCF0613": (
        "VCF0613",
        {
            1: "Agree",
            2: "Disagree",
            3: "Neither agree nor disagree (1988 and later only)",
        },
    ),
    "VCF0624": ("VCF0624", {1: "Not much", 2: "Some", 3: "A good deal"}),
    "VCF0803": (
        "VCF0803",
        {
            1: "Extremely liberal",
            2: "Liberal",
            3: "Slightly liberal",
            4: "Moderate, middle of the road",
            5: "Slightly conservative",
            6: "Conservative",
            7: "Extremely conservative",
        },
    ),
    "VCF0806": (
        "VCF0806",
        {
            1: "1 (Government insurance plan)",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7 (Private insurance plan)",
        },
    ),
    "VCF0819": (
        "VCF0819",
        {
            1: (
                "White people have a right to keep Negroes out of their neighborhoods i"
                "f they want to"
            ),
            2: (
                "Negroes have a right to live wherever they can aﬀord to, just like any"
                "body else"
            ),
        },
    ),
    "VCF0823": (
        "VCF0823",
        {
            1: "Agree (1956-1960: incl. ‘agree strongly’ and ‘agree but not strongly’)",
            2: (
                "Disagree (1956-1960: incl. ‘disagree strongly’ and ‘disagree but not s"
                "trongly’)"
            ),
        },
    ),
    "VCF0825": (
        "VCF0825",
        {1: "Less likely", 2: "About the same chances", 3: "More likely"},
    ),
    "VCF0830": (
        "VCF0830",
        {
            1: "1 (Government should help minority groups/blacks)",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7 (Minority groups/ blacks should help themselves)",
        },
    ),
    "VCF0838": (
        "VCF0838",
        {
            1: "By law, abortion should never be permitted.",
            2: (
                "The law should permit abortion only in case of rape, incest, or when t"
                "he woman’s life is in danger."
            ),
            4: (
                "By law, a woman should always be able to obtain an abortion as a matte"
                "r of personal choice."
            ),
        },
    ),
    "VCF0839": (
        "VCF0839",
        {
            1: "1 (Government should provide many fewer services: reduce spending a lot)",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7 (Government should provide many more services: increase spending a lot)",
        },
    ),
    "VCF0842": (
        "VCF0842",
        {
            1: "1 (Much tougher regulations to protet the environment)",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7 (Regulations already too much of a burden on business)",
        },
    ),
    "VCF0843": (
        "VCF0843",
        {
            1: "1 (Greatly decrease defense spending)",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7 (Greatly increase defense spending.)",
        },
    ),
    "VCF0846": ("VCF0846", {1: "Yes, important", 2: "Little to no importance"}),
    "VCF0850": (
        "VCF0850",
        {
            1: (
                "The Bible is the actual Word of God and is to be taken literally, word"
                " for word"
            ),
            2: (
                "The Bible is the Word of God but not everything in it should be taken "
                "literally, word for word"
            ),
            3: "The Bible is a book written by men and is not the Word of God",
        },
    ),
    "VCF0852": (
        "VCF0852",
        {
            1: "Agree strongly",
            2: "Agree somewhat",
            3: "Neither agree nor disagree",
            4: "Disagree somewhat",
            5: "Disagree strongly",
        },
    ),
    "VCF0853": (
        "VCF0853",
        {
            1: "Agree strongly",
            2: "Agree somewhat",
            3: "Neither agree nor disagree",
            4: "Disagree somewhat",
            5: "Disagree strongly",
        },
    ),
    "VCF0870": ("VCF0870", {1: "Better", 3: "Stayed same", 5: "Worse"}),
    "VCF0871": (
        "VCF0871",
        {
            1: "Much better",
            2: "Somewhat better",
            3: "Stayed same",
            4: "Somewhat worse",
            5: "Much worse",
        },
    ),
    "VCF0872": ("VCF0872", {1: "Get better", 3: "Stay about the same", 5: "Get worse"}),
    "VCF0876": ("VCF0876", {1: "Favor", 5: "Oppose"}),
    "VCF0876a": (
        "VCF0876a",
        {
            1: "Favor strongly",
            2: "Favor not strongly",
            4: "Oppose not strongly",
            5: "Oppose strongly",
        },
    ),
    "VCF0878": ("VCF0878", {1: "Yes", 5: "No"}),
    "VCF0880": (
        "VCF0880",
        {1: "Better Now", 2: "Same (2004: Volunteered)", 3: "Worse Now"},
    ),
    "VCF0880a": (
        "VCF0880a",
        {
            1: "Much better",
            2: "Somewhat better",
            3: "Same",
            4: "Somewhat worse",
            5: "Much worse",
        },
    ),
    "VCF0881": (
        "VCF0881",
        {
            1: "Better oﬀ (1956-1960,1964: get better)",
            2: "Same (1956-1960,1964: stay the way it is)",
            3: "Worse oﬀ (1956-1960,1964: get worse)",
        },
    ),
    "VCF0886": (
        "VCF0886",
        {1: "Increased", 2: "Same", 3: "Decreased (before 2012: or cut out entirely)"},
    ),
    "VCF0888": (
        "VCF0888",
        {1: "Increased", 2: "Same", 3: "Decreased (before 2012: or cut out entirely)"},
    ),
    "VCF0890": (
        "VCF0890",
        {1: "Increased", 2: "Same", 3: "Decreased (before 2012: or cut out entirely)"},
    ),
    "VCF0894": (
        "VCF0894",
        {1: "Increased", 2: "Same", 3: "Decreased (before 2012: or cut out entirely)"},
    ),
    "VCF0991": ("VCF0991", {1: "Approve", 5: "Disapprove"}),
}
ITEM_VARS = list(ITEM_QUESTIONS.keys())
YEAR_VAR = "VCF0004"
AGE_VAR = "VCF0101"
SEX_VAR = "VCF0104"
RACE_VAR = "VCF0105b"
REGION_VAR = "VCF0112"
EDU_VAR = "VCF0110"
SEX_LABELS = {1: "man", 2: "woman"}
RACE_LABELS = {1: "White", 2: "Black", 3: "Hispanic", 4: "another or multiple races"}
REGION_LABELS = {1: "the Northeast", 2: "the Midwest", 3: "the South", 4: "the West"}
EDU_LABELS = {
    1: "grade school or less",
    2: "high school",
    3: "some college",
    4: "a college or advanced degree",
}
AGE_BUCKET_EDGES = [16, 30, 45, 60, 200]
AGE_BUCKET_LABELS = ["16-29", "30-44", "45-59", "60+"]


def age_bucket_of(age: float) -> str:
    """Returns the age-bucket label for a respondent age.

    Args:
        age: Respondent age in years.

    Returns:
        Age-bucket label from AGE_BUCKET_LABELS. Ages outside the edges fall
        through to the last label.
    """
    for lo, hi, label in zip(
        AGE_BUCKET_EDGES[:-1],
        AGE_BUCKET_EDGES[1:],
        AGE_BUCKET_LABELS,
        strict=False,
    ):
        if lo <= age < hi:
            return label
    return AGE_BUCKET_LABELS[-1]


def cell_id_of(row: pd.Series) -> str:
    """Builds the demographic-group identifier for a respondent row.

    Args:
        row: Respondent row whose CELL_VARS columns are already labels.

    Returns:
        Pipe-joined group identifier.
    """
    return "|".join(str(row[var]) for var in CELL_VARS)


def build_persona(row: pd.Series) -> str:
    """Builds the persona sentence for a respondent's demographic group.

    Args:
        row: Respondent row whose CELL_VARS columns are already labels.

    Returns:
        Persona text placed before the question.
    """
    return (
        f"It is the {row['decade']}s. You are a {row['age']} year-old "
        f"{row['sex']} living in "
        f"{row['region']} of the United States. You identify as {row['race']} "
        f"and your highest "
        f"level of education is {row['education']}."
    )


__all__ = [
    "ANSWER_LETTERS",
    "SEED",
    "build_chat_messages",
    "build_question_block",
    "DATASET",
    "CELL_VARS",
    "ITEM_QUESTIONS",
    "ANES_ITEM_CODES",
    "ITEM_VARS",
    "age_bucket_of",
    "cell_id_of",
    "build_persona",
]
