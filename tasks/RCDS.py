###############################################################################
# Imports
import os
import pandas as pd
import random
import json

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

import sys
sys.path.append("../utils/") 
from utils.utils import process_responses_extraction, get_few_shot_prompt

###############################################################################

###############################################################################
# Constants
LABEL2ID = {
    "EMPTY": -1,  # Special label for empty responses
    "O": 0,
    "COMM": 1,
}
LABELS = list(LABEL2ID.keys())


# Columns to be logged together with the model's raw responses
# Note: DharmaBench uses 'id', 'root_file', 'comm_file' - removed 'file_name' and using 'passage' instead of 'text'
LOG_COLUMNS = ["id", "root_file", "comm_file", "root", "passage", "ground_truth", "offsets"]

###############################################################################


###############################################################################

# Prompts
PROMPTS = {
    "system": """You are a computational linguist and philologist specializing in matching Sanskrit root texts with their commentaries.
Your task is to identify a commentary on a given Sanskrit root text.

Definitions:
- Root Text: The original Sanskrit text that may be commented on.
- Passage: A Commentary passage of Sanskrit text that introduces, explains, glosses, and/or otherwise interprets the given root text.

Guidelines:
- Identify the span from the Root Text, from beginning to end, of commentary on the given Root Text in the provided Passage.
- Label each identified span as "COMM" (for commentary).
- There may be commentaries in the passage on other root texts; only identify the commentary for the given root text.

Return your output as a JSON with "prediction" key. The value is a list of dictionaries, each with:
- LABEL: "COMM"
- SPAN: the exact text span that contains the commentary (minimal required span)

Example for an item: {{"LABEL": "COMM", "SPAN": "vajreṇāṅkitam iti khaḍgasthaśrīkhaṇḍādhobhāge | evaṃ sarvatra karttikāyās tu muṣtyardhabhāge ||"}}

If no commentary is found, return an empty list under the "prediction" key.
Only respond with the JSON output, do not include any additional text or explanations.
""",
"user": "Root Text:\n{root}\n\nPassage:\n{passage}\n",
}

# Return your output as a JSON with "prediction" key. The value is a list of lists [LABEL, SPAN]:
# - LABEL: "COMM"
# - SPAN: the exact text span that contains commentary
#
# Only respond with the JSON output, do not include any additional text or explanations.
# """,
#
#     "user": "Root Text:\n{root}\n\nPassage:\n{passage}\n",
# }
# - If no relevant commentary is found, return an empty list

FEW_SHOT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("human", "- Root text:\n{root}\n\n- Passage:\n{passage}\n"),
        ("ai", "{output}"),
    ]
)


FEW_SHOT_EXAMPLES = [
    {
        "root": """svadevatākāraviśeṣaśūnyaṃ
prāg eva sambhāvya sukhaṃ sphuṭaṃ sat |
mahāsukhākhyaṃ jagadarthakāri
cintāmaṇiprakhyam uvāca kaścit || 9 ||""",
        "passage": """iti || tasya mañjuśrījñānasattvasya pañcajñānātmakasya saṃbandhinī yā nāmasaṃgītis tāṃ cāhaṃ dhārayiṣyāmīti ||

kiṃviśiṣṭām ity āha || gambhīrārthām ityādi | gambhīraś cāsav arthaś ca gambhīrārthaḥ śūnyatārthaḥ sa yasyām asti sā gambhīrārthā | atas tām dhārayiṣyāmīti || udārārthām iti | udāraś cāsāv arthaś codārārtho vaipulyārtha iti yāvat | sa yasyāṃ vidyate sā udārārthā | tāṃ ca || mahārthām iti | mahān artho yasyāṃ sā mahārthā | mahārthatvaṃ punar niravaśeṣasattvārthatā sarvāsāparipūraṇatvāt | tāṃ ca || asamām iti | na vidyate samā yasyā sāsamā | asamatvaṃ kutaḥ | dharmadhātusvabhāvatvāt | tāṃ ca || śivām iti sarvaprapañcopaśamatvāt || ādimadhyāntakalyāṇīm iti | śrutacintābhāvanākāleṣu harṣaprītipraśrabdhilābhāt trikalyāṇinī | tāṃ dhārayiṣyāmi || nāmasaṃgītim uttamām iti gatārtham ||

punar api kiṃviśiṣṭam ity āha | yātītair ityādi |""",
        "output": {
            "prediction": [
            ]
        },
    },

    {
        "root": """svadevatākāraviśeṣaśūnyaṃ
prāg eva sambhāvya sukhaṃ sphuṭaṃ sat |
mahāsukhākhyaṃ jagadarthakāri
cintāmaṇiprakhyam uvāca kaścit || 9 ||""",
        "passage": """sarva eva bhallābha iti | vyākhyātaprakṛṣṭabhaktiśālinām ayamāhlādaduḥkhamohairūpalakṣito loke yaḥ saṃvinmārgaḥ nīlapītādibodharūpaḥ panthāḥ sthitaḥ sa sarva eva tvatprāptihetuḥ | vedyasopānanimajjanakrameṇa paramavedakabhūmilābhāt |

he svāmin tavcchaktipātasamāveśamayabhaktyānandāsvādam anāsādya, bodhasya parā dehapātaprāpyā prakṛṣṭāpi yā śāntaśivapadātmā daśā syāt kaiścit sambhāvyate, sā taiḥ sambhāvyamānā māṃ praty āsavasya yathā śuktatā paryuṣitatā tathā bhātīti yāvat | yatas tair bhaktyamṛtam anāsvādyaiva śuktīkṛtam | yaiḥ punar āsvādyate, taiḥ svacamatkārānandaviśrāntīkṛtatvāt kā śuktatāsambhāvanā? āsvādād iti lyablope pañcamī | athavā tvadbhaktyamṛtāsvādād api parā mokṣarūpā yā kācid daśāstīti sambhāvyate, sā mahyaṃ na rocate, bhaktyamṛtāsvādasyaiva niratiśayacamatkāravattvād ity evaṃparam etat |

bhavadbhakti iti | vidyāvidyobhayasyāpīti vidyāvidyālakṣaṇasyobhayasya |""",
        "output": {
            "prediction": [
            ]
        },
    },

    {
        "root": """svadevatākāraviśeṣaśūnyaṃ
prāg eva sambhāvya sukhaṃ sphuṭaṃ sat |
mahāsukhākhyaṃ jagadarthakāri
cintāmaṇiprakhyam uvāca kaścit || 9 ||""",
        "passage": """yadi rādhikāyāṃ mānasaṃ manaḥ lagnasamādhi viṣayāsaṅge satyapi | hanteti harṣaviṣādayoḥ tarhi virahasya vyādhiḥ pīḍā | kathaṃ viṣayāsaṅgaḥ | iti anena prakāreṇa | adhunāpi | dṛśornetrayorvibhramāḥ vilāsāḥ svayameva tajjātīyā eva | kimbhūtāḥ | taralāḥ snigdhāḥ taralāśca te snigdhāśceti | yadatra svabhāvasaṃpratyayaḥ | vaktrāmbujasaurabhaṃ saugandhyaṃ yena sa ca | athāpi | girāṃ vācāṃ vakrimā vakrabhāvaḥ sa eva vartate | sudhāsyandī sudhāṃ syandituṃ śīlamasyeti sa tathā | yāsvāditā sapadi bimbādharamādhurī saiva tajjājīyaivāsvādyate | madhurasya bhāvaḥ mādhuryam | bimbādharasya mādhurī sā iti viśeṣā jñeyā |""",
        "output": {
            "prediction": [
                ('COMM', 'yadi rādhikāyāṃ mānasaṃ manaḥ lagnasamādhi viṣayāsaṅge satyapi | hanteti harṣaviṣādayoḥ tarhi virahasya vyādhiḥ pīḍā | kathaṃ viṣayāsaṅgaḥ | iti anena prakāreṇa | adhunāpi | dṛśornetrayorvibhramāḥ vilāsāḥ svayameva tajjātīyā eva | kimbhūtāḥ | taralāḥ snigdhāḥ taralāśca te snigdhāśceti | yadatra svabhāvasaṃpratyayaḥ | vaktrāmbujasaurabhaṃ saugandhyaṃ yena sa ca | athāpi | girāṃ vācāṃ vakrimā vakrabhāvaḥ sa eva vartate | sudhāsyandī sudhāṃ syandituṃ śīlamasyeti sa tathā | yāsvāditā sapadi bimbādharamādhurī saiva tajjājīyaivāsvādyate | madhurasya bhāvaḥ mādhuryam | bimbādharasya mādhurī sā iti viśeṣā jñeyā |')
            ]
        },
    },
    {
        "root": """svadevatākāraviśeṣaśūnyaṃ
prāg eva sambhāvya sukhaṃ sphuṭaṃ sat |
mahāsukhākhyaṃ jagadarthakāri
cintāmaṇiprakhyam uvāca kaścit || 9 ||""",
        "passage": """dākṣiṇyabhaṅgabhayān nīcamārgeṇa nirgatya punaḥ sarvādhyakṣa iva sañcarati tad vaditi dhvaniḥ ||

tvām iti || āmrāś cūtāḥ kūṭeṣu śikhareṣu yasya sa āmrakūṭo nāma sānumān parvataḥ || "āmraś cūto rasālo 'sau" iti | "kūṭo 'strī śikharaṃ śṛṅgam" iti cāmaraḥ || āsāro dhārāvṛṣṭiḥ || "dhārāsaṃpāta āsāraḥ" ity amaraḥ || tena praśamito vanopaplavo dāvāgnir yena tam | kṛtopakāram ity arthaḥ | adhvaśrameṇa parigataṃ vyāptaṃ tvāṃ sādhu samyak mūrdhnā vakṣyati voḍhā || baher ḷṛṭ || tathā hi kṣudraḥ kṛpaṇo 'pi || "kṣudro daridre kṛpaṇe nṛśaṃse" iti yādavaḥ || saṃśrayāya saṃśrayaṇāya mitre suhṛdi || "athaḥ mitraṃ sakhā suhṛt" ity amaraḥ || prāpta āgate sati | prathamasukṛtāpekṣayā pūrvopakāraparyālocanayā vimukho na bhavati | yas tathā tena prakāregoccair unnataḥ sa āmrakūṭaḥ kiṃ punarvimukho na bhavatīti kimu vaktavyam ity arthaḥ || etena prathamāvasathe saukhyalābhāt te kāryasiddhir astīti sūcitam | tad uktaṃ nimittanidāne—"prathamāvasathe yasya saukhyaṃ tasyākhile 'dhvani | śivaṃ bhavati yātrāyām anyathā tv aśubhaṃ dhruvam" || iti ||

channeti || he megha, pariṇataiḥ paripakvaiḥ phalair dyotanta iti tathoktaiḥ | āṣāḍhe vanacūtāḥ phalanti pacyanteḥ ca meghavātenety āśayaḥ | kānanāmrair vanacūtaiś channopānta āvṛtapārśve 'cala āmrakūṭādriḥ snigdhaveṇīsavarṇe masṛṇakeśabandhacchāye | śyāmavarṇaṃ ity arthaḥ || "veṇī tu keśabandhe jalasrutau" iti yādavaḥ | tvayi śikharaṃ śṛṅgam ārūḍhe sati || "yasya ca bhāvena bhāvalakṣaṇam" iti saptamī ||""",
        "output": {
            "prediction": [
                ('COMM', 'tvām iti || āmrāś cūtāḥ kūṭeṣu śikhareṣu yasya sa āmrakūṭo nāma sānumān parvataḥ || "āmraś cūto rasālo \'sau" iti | "kūṭo \'strī śikharaṃ śṛṅgam" iti cāmaraḥ || āsāro dhārāvṛṣṭiḥ || "dhārāsaṃpāta āsāraḥ" ity amaraḥ || tena praśamito vanopaplavo dāvāgnir yena tam | kṛtopakāram ity arthaḥ | adhvaśrameṇa parigataṃ vyāptaṃ tvāṃ sādhu samyak mūrdhnā vakṣyati voḍhā || baher ḷṛṭ || tathā hi kṣudraḥ kṛpaṇo \'pi || "kṣudro daridre kṛpaṇe nṛśaṃse" iti yādavaḥ || saṃśrayāya saṃśrayaṇāya mitre suhṛdi || "athaḥ mitraṃ sakhā suhṛt" ity amaraḥ || prāpta āgate sati | prathamasukṛtāpekṣayā pūrvopakāraparyālocanayā vimukho na bhavati | yas tathā tena prakāregoccair unnataḥ sa āmrakūṭaḥ kiṃ punarvimukho na bhavatīti kimu vaktavyam ity arthaḥ || etena prathamāvasathe saukhyalābhāt te kāryasiddhir astīti sūcitam | tad uktaṃ nimittanidāne—"prathamāvasathe yasya saukhyaṃ tasyākhile \'dhvani | śivaṃ bhavati yātrāyām anyathā tv aśubhaṃ dhruvam" || iti ||')
            ]
        },
    },

#     {
#         "root": """svadevatākāraviśeṣaśūnyaṃ
# prāg eva sambhāvya sukhaṃ sphuṭaṃ sat |
# mahāsukhākhyaṃ jagadarthakāri
# cintāmaṇiprakhyam uvāca kaścit || 9 ||""",
#         "passage": """pumvapuṣīti | prajñāsamparkonmukhe iti bhāvaḥ | hitaṃ tatkathanaṃ pañcākāreṇeti | nirmitādhārādheyaprapañcarūpeṇautpattikrameṇa ādikarmikāṇām etadvyātirekeṇa kathayitum aśakyatvād iti bhāvaḥ ||

# athetyādi | sarvastrīṣu mahākaruṇām āmukhīkṛtyāta eva dveṣavajrīsamādhiṃ samāpadyedam udājahāra | śūnyatā viramānandaḥ | karuṇā ānandatrayaṃ tasyām | abhinnā kevalamahāsukhasvabhāvety arthaḥ | ata eva divyakāmasukhe na sthitā vikalpa ānandādiprabhedakalpanā | prapañco bījacihnādivikalpaḥ | nirākulā cittaikāgratayā ||

# nāryaḥ striyaḥ | sarvastrīṇāṃ dehaḥ puruṣasamparkonmukhaḥ tasmin sthitā ||""",
#         "output": {
#             "prediction": [
#                 ('COMM', 'athetyādi | sarvastrīṣu mahākaruṇām āmukhīkṛtyāta eva dveṣavajrīsamādhiṃ samāpadyedam udājahāra | śūnyatā viramānandaḥ | karuṇā ānandatrayaṃ tasyām | abhinnā kevalamahāsukhasvabhāvety arthaḥ | ata eva divyakāmasukhe na sthitā vikalpa ānandādiprabhedakalpanā | prapañco bījacihnādivikalpaḥ | nirākulā cittaikāgratayā ||')
#             ]
#         },
#     },

#     {        
#         "root": """svadevatākāraviśeṣaśūnyaṃ
# prāg eva sambhāvya sukhaṃ sphuṭaṃ sat |
# mahāsukhākhyaṃ jagadarthakāri
# cintāmaṇiprakhyam uvāca kaścit || 9 ||""",
#         "passage": """kiṃsvabhāvāḥ | hṛṣṭatuṣṭāśayasvabhāvāḥ | tatra hṛṣṭāḥ kāyais tuṣṭāś cittaiḥ | āśayo 'bhiprāyaḥ | tataś cāyam arthaḥ | hṛṣṭatuṣṭā āśayā yeṣāṃ te hṛṭṣatuṣṭāśayā ata āha hṛṣṭatuṣṭāśayair iti | ata eva muditā harṣitāḥ sarvasattvārthakaraṇāt | ata evāha muditair iti || vigrahasya rūpaṃ vigraharūpaṃ | krodhānāṃ vigraharūpaṃ krodhavigraharūpaṃ | kiṃ tadrūpaṃ mahābhairavāṭṭaṭṭahāsānekaśīrṣakaracaraṇavikṛtanayanordhvakeśapiṅgalabhrūbhaṅgakapālamālālaṃkṛtisarpābharaṇavyāghracarmaniveśanādikaṃ | tad vidyate yeṣāṃ te krodhavigraharūpinaḥ | ata āha krodhavigraharūpibhir iti || buddhānāṃ kṛtyāni buddhakṛtyāni kṛtyāni kartavyāni | ata āha buddhakṛtyakarair iti | ata eva nāthāḥ śāstāro buddhakṛtyakaraṇāt | ato nāthair iti || praṇato vigraho yeṣāṃ te praṇatavigrahāḥ | atas taiḥ | praṇatavigrahaiḥ sārdhaṃ bhagavān vajradhara idaṃ vakṣyamāṇam āheti saṃbandhaḥ ||""",
#         "output": {
#             "prediction": [
#             ]
#         },
#     },

#     {        
#         "root": """svadevatākāraviśeṣaśūnyaṃ
# prāg eva sambhāvya sukhaṃ sphuṭaṃ sat |
# mahāsukhākhyaṃ jagadarthakāri
# cintāmaṇiprakhyam uvāca kaścit || 9 ||""",
#         "passage": """avisarjanīyaṃ hi tricīvaramuktaṃ tathāgatena. sacec chāriputra bodhisattvas tricīvaraṃ parityajya yācanaguruko bhavet, na tena alpecchatā āsevitā bhavet. iti.

# atityāgaṃ niṣedhayan punar ātmarakṣām upadarśayann āha.

# satāṃ satpuruṣāṇāṃ bodhisattvānāṃ dharmaḥ. laukikalokottaraparahitasukhavidhānam | tatsevakaṃ kāyam alpārthanimittaṃ na pīḍayet. anyathā mahato 'rtharāśer hāniḥ syāt. ata eva pūrvasmin hetupadam etat. kutaḥ punar evam? yasmād anenaiva sukumāropakrameṇa saṃvardhamānaḥ śīghram eva sattvānāṃ hitasukhasaṃpādanasamartho bhavati.


# yata evaṃ tasmāt svaśarīraśirodānādi na kartavyam iti niṣiddham. kadā?""",
#         "output": {
#             "prediction": [
#             ]
#         },
#     },

#     {
#         "root": """svadevatākāraviśeṣaśūnyaṃ
# prāg eva sambhāvya sukhaṃ sphuṭaṃ sat |
# mahāsukhākhyaṃ jagadarthakāri
# cintāmaṇiprakhyam uvāca kaścit || 9 ||""",
#         "passage": """durvijñeyā hi sāvasthā kim apy etad anuttamam || iti |

# prakārāntareṇāpi tad evāha—śivetyādi | ratnāntargatasaṃsthitam iti maṇimadhyastham ||

# viditaitajjñānasya yogino niṣṭhām āha—jñānāmṛtetyādi | subodham ||""",
#         "output": {
#             "prediction": [
#                 ('COMM', 'prakārāntareṇāpi tad evāha—śivetyādi | ratnāntargatasaṃsthitam iti maṇimadhyastham ||')
#             ]
#         },
#     },
]


###############################################################################


###############################################################################
# Functions

def get_data(data_dir, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = os.path.join(data_dir, "Sanskrit/RCDS/test.jsonl")
    train = os.path.join(data_dir, "Sanskrit/RCDS/train.jsonl")
    
    if os.path.exists(test):
        data = pd.read_json(test, lines=True)
        if os.path.exists(train):
            train_data = pd.read_json(train, lines=True)
        else:
            train_data = pd.DataFrame(columns=data.columns)
        
        if 'passage' in data.columns and 'text' not in data.columns:
            data['text'] = data['passage']
        if not train_data.empty and 'passage' in train_data.columns and 'text' not in train_data.columns:
            train_data['text'] = train_data['passage']

    data["ground_truth"] = data["ground_truth"].apply(lambda x: [(item[0], item[1]) for item in x] if isinstance(x, list) else [])
    data["offsets"] = data["offsets"].apply(lambda x: [(item[0], item[1]) for item in x] if isinstance(x, list) else [])

    if not train_data.empty:
        train_data["ground_truth"] = train_data["ground_truth"].apply(lambda x: [(item[0], item[1]) for item in x] if isinstance(x, list) else [])
        train_data["offsets"] = train_data["offsets"].apply(lambda x: [(item[0], item[1]) for item in x] if isinstance(x, list) else [])
    
    return train_data, data


def get_prompt(config: dict, **kwargs) -> ChatPromptTemplate:
    """
    Get the prompt for the given task and prompt type.
    Also returns the schema for the task if model supports structured output.
    :param config: configuration
    :param kwargs: keyword arguments
    :return: prompt
    """
    # Get kwargs
    prompt_type = config["prompt_type"]
    seed = config["seed"]

    system_prompt = PROMPTS["system"]
    user_prompt = PROMPTS["user"]

    messages = [("system", system_prompt), ("human", user_prompt)]

    # Add few-shot examples to the prompt
    if "few_shot" in prompt_type:
        random.seed(seed)  # Set the seed for reproducibility
        examples = FEW_SHOT_EXAMPLES
        random.shuffle(examples)

        # Make sure samples are in json format, created from dictionary
        json_examples = []
        for example in examples:
            json_examples.append({
                "root": example["root"],
                "passage": example["passage"],
                "output": json.dumps(example["output"], ensure_ascii=False)
            })

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=FEW_SHOT_PROMPT_TEMPLATE,
            examples=json_examples,
        )

        # Add in position 1 (after the system prompt)
        messages.insert(1, few_shot_prompt)

    prompt = ChatPromptTemplate.from_messages(messages)

    return prompt

def get_user_inputs(data: pd.DataFrame) -> list[dict]:
    """
    Get user inputs for the given task.
    :param data: data
    :return: user inputs as a list of dictionaries
    """
    user_inputs = [
        {"root": row["root"], "passage": row["passage"]}
        for _, row in data.iterrows()
    ]
    return user_inputs


process_responses = process_responses_extraction


###############################################################################

###############################################################################
# Export
RCDS = {
    "get_data": get_data,
    "get_prompt": get_prompt,
    "get_user_inputs": get_user_inputs,
    "label2id": LABEL2ID,
    "process_responses": process_responses,
    "columns_to_log": LOG_COLUMNS,
}
