"""GSS questions, demographic groups, and prompts for Experiment 2."""

from __future__ import annotations

import string

import pandas as pd

SEED = 42
DATASET = "gss"
CELL_VARS = ["year", "age", "sex", "race", "region", "degree"]
ORIGINAL_7_GSS_VARS = [
    "cappun",
    "gunlaw",
    "natfare",
    "happy",
    "satfin",
    "polviews",
    "partyid",
]
ITEM_QUESTIONS = {
    "polviews": (
        "Where would you place yourself on a scale of political views from extremely li"
        "beral to extremely conservative?"
    ),
    "natspac": "space exploration program",
    "natenvir": "improving & protecting environment",
    "natheal": "improving & protecting nations health",
    "natcity": "solving problems of big cities",
    "natcrime": "halting rising crime rate",
    "natdrug": "dealing with drug addiction",
    "nateduc": "improving nations education system",
    "natrace": "improving the conditions of black people",
    "natarms": "military, armaments, and defense",
    "nataid": "foreign aid",
    "natfare": (
        "Is the United States spending too much, too little, or about the right amount "
        "on welfare?"
    ),
    "natroad": "highways and bridges",
    "natsoc": "social security",
    "natmass": "mass transportation",
    "natpark": "parks and recreation",
    "natchld": "assistance for childcare",
    "natsci": "supporting scientific research",
    "natenrgy": "developing alternative energy sources",
    "natspacy": "space exploration:version y",
    "natenviy": "the environment:version y",
    "nathealy": "health:version y",
    "natcityy": "assistance to big cities:version y",
    "natcrimy": "law enforcement:version y",
    "natdrugy": "drug rehabilitation:version y",
    "nateducy": "education:version y",
    "natracey": "assistance to black people: version y",
    "natarmsy": "national defense:version y",
    "nataidy": "assistance to other countries:ver y",
    "natfarey": "assistance to the poor:version y",
    "eqwlth": "should govt reduce income differences",
    "tax": "r's federal income tax",
    "cappun2": "does r favor capital punishment",
    "cappun": "Do you favor or oppose the death penalty for persons convicted of murder?",
    "gunlaw": (
        "Would you favor or oppose a law requiring a person to obtain a police permit b"
        "efore buying a gun?"
    ),
    "courts": "courts dealing with criminals",
    "wirtap": "wiretapping",
    "grass": "should marijuana be made legal",
    "fund": "how fundamentalist is r currently",
    "postlf1": "life of peace and tranquility",
    "postlf2": "life of intense action",
    "postlf3": "like here on earth only better",
    "postlf4": "life without many earthly joys",
    "postlf5": "pale or shadowy form of life",
    "postlf6": "a spiritual life involving mind not body",
    "postlf7": "paradise of pleasures and delights",
    "postlf8": "place of loving intellectual communion",
    "postlf9": "union with god",
    "postlf10": "reunion with loved ones",
    "judge": "image of god as judge",
    "redeemer": "image of god as redeemer",
    "lover": "image of god as lover",
    "master": "image of god as master",
    "mother": "image of god as mother",
    "creator": "image of god as creator",
    "father": "image of god as father",
    "spouse": "image of god as spouse",
    "friend": "image of god as friend",
    "king": "image of god as king",
    "liberatr": "image of god as liberator",
    "healer": "image of god as healer",
    "popespks": "pope is infallible on matter's of faith or morals",
    "fund16": "how fundamentalist was r at age 16",
    "spfund": "how fundamentalist is spouse currently",
    "spfund16": "how fundamentalist was spouse at 16",
    "prayer": "bible prayer in public schools",
    "racdin": "object to opp. race home for dinner",
    "racpush": "blacks shouldn't push",
    "racseg": "whites have right to seg. neighborhood",
    "busing": "attitude toward racial busing",
    "affrmact": "favor preference in hiring black people",
    "happy": (
        "Taking all things together, how would you say things are these days — would yo"
        "u say you are very happy, pretty happy, or not too happy?"
    ),
    "hapmar": "happiness of marriage",
    "health": "condition of health",
    "satcity": "city or place r lives in",
    "sathobby": "non-working activities, hobbies",
    "satfam": "family life",
    "satfrnd": "friendships",
    "sathealt": "health and physical condition",
    "confinan": "confid in banks & financial institutions",
    "conbus": "confidence in major companies",
    "conclerg": "confidence in organized religion",
    "coneduc": "confidence in education",
    "confed": "confid. in exec branch of fed govt",
    "conlabor": "confidence in organized labor",
    "conpress": "confidence in press",
    "conmedic": "confidence in medicine",
    "contv": "confidence in television",
    "conjudge": "confid. in united states supreme court",
    "consci": "confidence in scientific community",
    "conlegis": "confidence in congress",
    "conarmy": "confidence in military",
    "aged": "should aged live with their children",
    "anomia1": "next to health, money is most important",
    "anomia2": "wonder if anything is worthwhile",
    "anomia3": "no right & wrong ways to make money",
    "anomia4": "live only for today",
    "anomia5": "lot of the average man getting worse",
    "anomia6": "not fair to bring child into world",
    "anomia7": "officials not interested in average man",
    "anomia8": "don't know whom to trust",
    "anomia9": "most don't care what happens to others",
    "joblose": "is r likely to lose job",
    "satjob": "work satisfaction",
    "satfin": "How satisfied are you with your present financial situation?",
    "fehome": "women take care of home not country",
    "fework": "should women work",
    "fepol": "women not suited for politics",
    "abfirm": "how firm is r's opinion on abortion",
    "pillok": "birth control to teenagers 14-16",
    "sexeduc": "sex education in public schools",
    "premarsx": "sex before marriage",
    "teensex": "sex before marriage:teens 14-16",
    "xmarsex": "sex with person other than spouse",
    "homosex": "homosexual sex relations",
    "pornlaw": "feelings about pornography laws",
    "spanking": "favor spanking to discipline child",
    "fechld": "mother working doesn't hurt children",
    "fehelp": "wife should help husbands career first",
    "fepresch": "preschool kids suffer if mother works",
    "fefam": "better for man to work, woman tend home",
    "feserve": "mandatory conscription for females?",
    "meserve": "mandatory conscription for males?",
    "milqual": "quality of persons in military",
    "milpay": "pay and benefits for military personnel",
    "fenumok": "number of women in military",
    "hinumok": "number of hispanics in military",
    "blnumok": "number of blacks in military",
    "draftfem": "if emergency, draft women also?",
    "helppoor": "should govt improve standard of living?",
    "helpnot": "should govt do more or less?",
    "helpsick": "should govt help pay for medical care?",
    "helpblk": "should govt aid black people?",
    "punsin": "r agrees that sinners must be punished",
    "blkwhite": "r agrees right & wrong not black & white",
    "rotapple": "r agrees immoral person corrupts society",
    "permoral": "r agrees that morality a personal matter",
    "ownthing": "youthful protest is good",
    "liveblks": "neighborhood half black",
    "livewhts": "r favors living in half white neighborhood",
    "marblk": "close relative marry black person",
    "marasian": "close relative marry asian",
    "marhisp": "close relative marry hispanic",
    "marwht": "r favor close relative marrying white person",
    "discaff": "whites hurt by aff. action",
    "selfirst": "take care of yourself first before others",
    "judgeart": "only a few people can judge art",
    "trstprof": "trust profs judgments in assigning reading",
    "modpaint": "modern painting is just slapped on",
    "pclit": "shame tradl amer lit ignored for minority lit",
    "fejobaff": "for or against preferential hiring of women",
    "discaffm": "a man won't get a job or promotion",
    "discaffw": "a woman won't get a job or promotion",
    "fehire": "should hire and promote women",
    "rellife": "try to carry beliefs into other dealings",
    "satdemoc": "r satisfied with way democracy works in america",
    "ethimp": "importance of ethnic group to r",
    "ethignor": "us harmony by ignore ethnic differences",
    "ethnofit": "eth minorities never fit in am mainstream",
    "ethtrads": "society recognize right of ethnic traditions",
    "ethadapt": "minorities must adapt to american culture",
    "ethsame": "individual of same ethnic group similar",
    "ethdiff": "ethnic groups in us different",
    "whtsdiff": "whites different from minorities",
    "docaskme": "r prefers doc ask opinion",
    "docdecid": "r leaves decisions up to doctor",
    "docrely": "r relies on doctors knowledge",
    "fininc1": "importance to r to be told about incentives",
    "fininc4": "r likely to ask doc about incentives",
    "docsat": "how satisfied r with office visit",
    "learnnew": "job requires r to learn new things",
    "workfast": "job requires r to work fast",
    "workdiff": "r does numerous things on job",
    "lotofsay": "r has lot of say in job",
    "overwork": "r has too much work to do well",
    "knowwhat": "r knows what's expected on job",
    "myskills": "job allows r use of skills",
    "respect": "r treated with respect at work",
    "trustman": "r trust management at work",
    "safetywk": "worker safety priority at work",
    "safefrst": "no shortcuts on worker safety",
    "teamsafe": "mgt and employees work together re safety",
    "safehlth": "safety and health condition good at work",
    "proudemp": "r proud to work for employer",
    "prodctiv": "work conditions allow productivity",
    "trdunion": "workers need strong unions",
    "promteok": "r's chances for promotion good",
    "opdevel": "opportunity to develop my abilities",
    "hlpequip": "enough help and equip to ge the job done",
    "haveinfo": "enough info to get the job done",
    "fringeok": "fringe benefits are good",
    "supcares": "supervisor concerned about welfare",
    "condemnd": "r free from conflicting demands",
    "promtefr": "promotions are handled fairly",
    "jobsecok": "the job security is good",
    "suphelp": "supervisor helpful to r in getting job done",
    "trainops": "r have the training opportunities",
    "manvsemp": "relations bw management and employees",
    "trynewjb": "how likely r make effort for new job next year",
    "health1": "r's health in general",
    "spvtrfair": "supervisor is fair",
    "satjob1": "job satisfaction in general",
    "quallife": "r's quality of life",
    "hlthphys": "r's physical health",
    "hlthmntl": "r's mental health, mood, and ability to think",
    "satsoc": "r's satisfaction with social activities and relationships",
    "actssoc": "how well r carries out usual social activities",
    "abfelegl": "women only: women should be able to have legal abortions",
    "abmoral": "r has a moral opposition to abortion",
    "abinspay": "should people be able to use health insurance for abortion - ver y",
    "talkemp": "how likely to talk directly if coworker not work",
    "talksup": "how likely to speak supervisor if coworker not w",
    "donothng": "how likely to do nothing if coworker not working",
    "nextgen": "science & tech. give more opportunities to next generation",
    "toofast": "science makes our way of life change too fast",
    "advfront": "sci rsch is necessary and should be supported by federal govt",
    "balpos": "sci research is strongly in favor of benefits",
    "hotcore": "sci knowledge: the center of earth is very hot",
    "radioact": "sci knowledge: all radioactivity is man-made",
    "boyorgrl": "sci knowledge: father gene decides sex of baby",
    "lasers": "sci knowledge: lasers work by focusing sound waves",
    "electron": "sci knowledge: electrons are smaller than atoms",
    "viruses": "sci knowledge: antibiotics kill viruses as well as bacteria",
    "bigbang": "sci knowledge: the universe began with a huge explosion",
    "condrift": "sci knowledge: the continents have been moving",
    "evolved": "sci knowledge: human beings developed from animals",
    "scientgo": "scientists work for good of humanity",
    "scienthe": "scientists help solve prob",
    "scientod": "scientists odd and peculiar",
    "scientbe": "scientists want to make life better for avg person",
    "protest1": "allow public meetings protesting govt",
    "protest3": "allow demonstrations protesting govt",
    "protest6": "allow national anti-govt strike",
    "revspeak": "allow revolutionary public meetings",
    "revpub": "allow revolutionaries to publish books",
    "setwage": "govt control of wages",
    "setprice": "govt control of prices",
    "cutgovt": "cuts in govt spending",
    "makejobs": "govt create new jobs",
    "lessreg": "less govt regulation of business",
    "hlphitec": "govt support for new products and tech",
    "savejobs": "govt support for declining industries",
    "cuthours": "reduction in length of work week",
    "laborpow": "power of labor unions in r's country",
    "buspow": "power of business and industry",
    "govtpow": "power of federal govt",
    "jobsall": "govts resp: provide jobs for all",
    "pricecon": "govts resp: keep prices under control",
    "hlthcare": "govts resp: provide hlth care for sick",
    "aidold": "govts resp: provide for the elderly",
    "aidindus": "govts resp: assist industrial growth",
    "aidunemp": "govts resp: provide for the unemployed",
    "equalize": "govts resp: reduce income differences",
    "aidcol": "gov. should assist low-income college student",
    "aidhouse": "gov. should provide housing to poor",
    "poleff11": "don't have any say about what the government does",
    "healthissp": "merged healthissp_d and healthissp_e",
    "goodlife": "standard of living of r will improve",
    "goveqinc": "govmnt should reduce inc differentials",
    "famsuffr": "family life suffers if mom works f-t",
    "twoincs": "husb & wife should contrib to family income",
    "marsame": "homosexual couples should have the right to marry one another",
    "marsame1": "homosexual couples have right to marry",
    "cohabok": "living together as an acceptable option",
    "cohabfst": "people should live together before marriage",
    "happy7": "how happy r is",
    "satfam7": "family satisfaction in general",
    "hapunhap": "happy or unhappy with life today",
    "premars1": "is premarital sex wrong?",
    "xmarsex1": "is extramarital sex wrong?",
    "homosex1": "is homosexual sex wrong?",
    "abdefctw": "wrong for woman to get abortion for birth defects",
    "abpoorw": "wrong for woman to get abortion if low income?",
    "concong": "confidence in us congress",
    "conbiz": "confidence in business and industry",
    "conchurh": "confidence in churches and religious orgs",
    "concourt": "confidence in courts and legal system",
    "conschls": "confidence in schools and educ system",
    "clergvte": "clergy shouldn't try to influence voters",
    "clerggov": "clergy shouldn't try to influence govt decisions",
    "afterlif": "belief in life after death",
    "heaven": "belief in heaven",
    "theism": "god concerned with human beings personally",
    "fatalism": "people can't change the course of their lives",
    "godmeans": "life meaningful because god exists",
    "nihilism": "life serves no purpose",
    "egomeans": "life meaningful only if you provide meaning",
    "ownfate": "we each make our own fate",
    "religinf": "the u.s. would be a better country if religion had less influence",
    "privent": "private enterprise will solve u.s. problems",
    "scifaith": "believe too much in science, not enough faith",
    "harmgood": "modern science does more harm than good",
    "sciworse": "humans change nature for the worse",
    "scigrn": "modern science will solve envir probs",
    "grnecon": "worry too much about envir, too little econ",
    "harmsgrn": "almost everything we do harms envir",
    "anrights": "animals have rights too",
    "resnatur": "humans should respect naturecreated by god",
    "grnprog": "worry too much about progress harming envir",
    "naturpax": "nature would be at peace if humans left alone",
    "grwthelp": "amer needs econ grwth to protect envir",
    "antests": "ok to test on animals to save humans",
    "naturwar": "nature is survival of fittest",
    "grwtharm": "econ grwth always harms envir",
    "toodifme": "too diff to do anything about envir",
    "ihlpgrn": "do what i can to help envir",
    "scitest1": "humans make all radioactivity",
    "scitest2": "antibiotics kill bacteria not viruses",
    "scitest3": "astrology has some scientific truth",
    "scitest4": "humans evolved from animals",
    "scitest5": "all man-made chems cause cancer",
    "grntest1": "all will die if exposed to radioact",
    "grntest2": "nuke waste dangerous for thousands of yrs",
    "grntest3": "greenhouse effect caused by hole in atmosphere",
    "grntest4": "using coal or gas contrib to greenhouse effect",
    "grntest5": "all pesticides cause cancer",
    "grntest6": "humans are main cause of plant & animal extinct",
    "grntest7": "car's not imp cause of air pollution",
    "carsten": "car pollution will increase in next 10 yrs",
    "imports": "america should limit the import",
    "immcrime": "immigrants increase crime rates",
    "immjobs": "immigrants take jobs away",
    "govdook": "we can trust people in govt",
    "polgreed": "most politicians are only for what get out of politics",
    "genetst1": "know about genetic testing",
    "sectech": "should govt maintain secrecy over technology",
    "secdocs": "does govt classify too many documents as secrets",
    "askfinan": "govt investigate workers finances for security",
    "askcrime": "govt investigate workers crime records",
    "askdrugs": "govt investigate workers illegal drug use",
    "askmentl": "govt investigate workers mental health",
    "askforgn": "govt investigate workers foreign family",
    "askdrink": "govt investigate workers alcohol use",
    "asksexor": "govt investigate workers sexual orientation",
    "secdiplo": "government maintain secrecy on diplomatic initiatives",
    "secmilop": "government maintain secrecy on military operations",
    "secterr": "government maintain secrecy on domestic terrorism",
    "secbudgt": "government maintain secrecy on the intelligence budget",
    "chkfinan": "government should verify financial assets",
    "chkspfin": "government should verify spouse's financial assets",
    "chktaxes": "government should verify tax records",
    "religid": "r's religious identity",
    "hlthstrt": "would you say the respondent? health in general is excellent, good, fair, or poo",
    "rhlthend": "r health rating",
    "religimp": "how important is religion",
    "relidimp": "how important is r's religion",
    "relidins": "to what extent r feels insulted if someone criticized r's religion",
    "freemind": "to understand the world, we must free our minds from old traditions and beliefs",
    "decevidc": "when i make important decisions in my life, i rely mostly on reason and evidence",
    "advfmsci": "all of the greatest advances for humanity have come from science and technology",
    "prochoic": "i consider myself pro-choice.",
    "prolife": "i consider myself pro-life.",
    "godusa": "the success of the us is part of god’s plan",
    "govchrst": "the federal government should advocate christian values.",
    "hivafraid": "afraid to be around a person with hiv",
    "hivimmrl": "people with hiv have participated in immoral activities",
    "hivdscrm": "there is a lot of discrimination against people with hiv",
    "defund": "move funds for police to social service",
    "racesurv17": "how much, if at all, do you think the legacy of slavery affects the position of ",
    "conpharvacy": "confidence in pharmaceutical companies y",
    "confedvacy": "confidence in the federal government y",
    "courtsv": "courts dealing with criminals (with volunteered response on web)",
    "prayernv": "bible prayer in public schools (no volunteered response on web)",
    "courtsnv": "courts dealing with criminals (no volunteered response on web)",
    "discaffwnv": "a woman won't get a job or promotion (no volunteered response on web)",
    "grassnv": "should marijuana be made legal (no volunteered response on web)",
    "fepolnv": "women not suited for politics (no volunteered response on web)",
    "vietdraft": "born on date with a vietnam war draft lottery number",
    "svyenjoy": "enjoy taking surveys",
    "svyid1": "willingness to use video chat",
    "svyid2": "willingnes for in-person survey",
    "lkelyvot": "likelihood to vote in 2024 election",
    "econstat": "what do you think about the state of the economy these days in the united states",
    "chngdocs": "changing official documents",
    "whovotets": "voting intention - time series",
    "partyid": "Generally speaking, how would you describe your political party affiliation?",
}
ITEM_VARS = list(ITEM_QUESTIONS.keys())
ANSWER_LETTERS = list(string.ascii_uppercase)
AGE_BUCKET_EDGES = [18, 30, 45, 60, 120]
AGE_BUCKET_LABELS = ["18-29", "30-44", "45-59", "60+"]


def decade_of(year: int) -> str:
    """Returns the decade label for a survey year.

    Args:
        year: GSS survey year.

    Returns:
        Decade string such as ``1990s``.
    """
    return f"{int(year) // 10 * 10}s"


def age_bucket_of(age: float) -> str:
    """Returns the age-bucket label for a respondent age.

    Args:
        age: Respondent age in years.

    Returns:
        Age-bucket label from AGE_BUCKET_LABELS. Ages outside the edges
        fall through to the last label.
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
        row: Respondent row carrying the CELL_VARS columns.

    Returns:
        Pipe-joined group identifier. Year is coarsened to decade and age
        to its bucket.
    """
    parts = [
        decade_of(row["year"]),
        age_bucket_of(row["age"]),
        str(row["sex"]),
        str(row["race"]),
        str(row["region"]),
        str(row["degree"]),
    ]
    return "|".join(parts)


def build_persona(row: pd.Series) -> str:
    """Builds the persona sentence for a respondent's demographic group.

    Args:
        row: Respondent row carrying the CELL_VARS columns.

    Returns:
        Persona text placed before the question.
    """
    return (
        f"It is the {decade_of(row['year'])}. You are a "
        f"{age_bucket_of(row['age'])} year-old "
        f"{row['race']} {row['sex']} living in the {row['region']} region "
        f"of the United States. "
        f"Your highest education credential is: {row['degree']}."
    )


def build_question_block(question: str, options: list[str]) -> str:
    """Renders the question and lettered answer options.

    Args:
        question: Item question wording.
        options: Ordered answer-option labels.

    Returns:
        Question-and-options block ending with the answer instruction.
    """
    lines = [question, "", "Options:"]
    for letter, option in zip(ANSWER_LETTERS, options, strict=False):
        lines.append(f"  {letter}) {option}")
    lines.append("")
    lines.append(
        "Respond with only the single letter of the option that best matches your view."
    )
    return "\n".join(lines)


def build_chat_messages(persona: str, question: str, options: list[str]) -> list[dict]:
    """Builds the chat messages for one question under a persona.

    Args:
        persona: Persona text from build_persona.
        question: Item question wording.
        options: Ordered answer-option labels.

    Returns:
        A single user turn for the tokenizer chat template.
    """
    content = f"{persona}\n\n{build_question_block(question, options)}"
    return [{"role": "user", "content": content}]
