import unittest

from utils import get_responses_extraction_predictions, parse_response


class MyTestCase(unittest.TestCase):
    def test_get_responses_extraction_predictions_correct_1(self):
        raw_response = [{"raw_response":
            ["{\n    \"prediction\": [\n        (\"COMM\", \"svadevatākāraviśeṣaśūnyaṃ\\nprāg eva sambhāvya sukhaṃ sphuṭaṃ sat |\\nmahāsukhākhyaṃ jagadarthakāri\\ncintāmaṇiprakhyam uvāca kaścit || 9 ||\")\n    ]\n}"
            ]}]

        predictions, hallucinated  = get_responses_extraction_predictions(raw_response)
        print(predictions)

        self.assertEqual(1, len(hallucinated))
        self.assertEqual([False], hallucinated)
        self.assertEqual('COMM', predictions[0][0][0])

    def test_get_responses_extraction_predictions_incorrect_1(self):
        raw_response = [{"raw_response":[
            "{\n    \"prediction\": [\n        (\"COMM\", \"sadasator utpādāyogād\" ityādinājātaṃ jagad yena buddhaṃ pratyakṣato jñātaṃ tasya prtītyasamutpannā buddhiḥ śuddhaiva bodhato nirvedhataḥ | atas tasya dhīmato 'nābhogenābhogaṃ vinaiva nijaṃ sahajaṃ jagat satyam avitatham ||\")\n    ]\n}"
        ]}]

        predictions, hallucinated  = get_responses_extraction_predictions(raw_response)
        print(predictions)

        self.assertEqual(1, len(hallucinated))
        self.assertEqual(True, hallucinated[0])
        self.assertEqual([], [])

    # Handling extraction response that looks like - {'prediction': [{'LABEL': 'Author', 'SPAN': 'paṇinaḥ'}]}
    def test_get_responses_extraction_predictions_correct_label_span_dict(self):
        raw_response = [{"raw_response":[
            '{\n  "prediction": [\n    {\n      "LABEL": "Author",\n      "SPAN": "paṇinaḥ"\n    }]}'
        ]}]

        predictions, hallucinated  = get_responses_extraction_predictions(raw_response)
        #print(predictions)

        self.assertEqual(1, len(hallucinated))
        self.assertEqual(False, hallucinated[0])
        self.assertEqual('Author', predictions[0][0][0])
        self.assertEqual('paṇinaḥ', predictions[0][0][1])


    def test_classification_parse_response_correct_1(self):
        raw_response = "```json\n{\n  \"label\": \"TRUE\"\n}\n```"

        resp = parse_response(raw_response)

        self.assertTrue(resp and "label" in resp)
        self.assertEqual("TRUE", resp["label"])

    def test_classification_parse_response_incorrect_1(self):
        raw_response = "```json\n{\n  \"label\": \"FALSE\"\n}\n``` \n\nThe commentary provided does not directly address the root \"cittaṃ rakṣitukāmānāṃ mayaiṣa kriyate 'ñjaliḥ | smṛtiṃ ca saṃprajanyaṃ ca sarvayatnena rakṣata || 5.23 ||\". Instead, it elaborates on the concepts of smṛti (mindfulness) and saṃprajanya (clear comprehension), which are mentioned in the verse, but it does not provide a direct commentary on the root itself."

        resp = parse_response(raw_response)

        self.assertEqual({}, resp)

    def test_get_responses_extraction_predictions_empty_correct_1(self):
        raw_response = [{"raw_response":
            ["{\n    \"prediction\": []\n}"
            ]}]

        predictions, hallucinated  = get_responses_extraction_predictions(raw_response)
        print(predictions)

        self.assertEqual(1, len(hallucinated))
        self.assertEqual([False], hallucinated)

    def test_get_responses_extraction_predictions_empty_incorrect_2(self):
        raw_response = [{"raw_response":
            ["{}"
            ]}]

        predictions, hallucinated  = get_responses_extraction_predictions(raw_response)
        print(predictions)

        self.assertEqual(1, len(hallucinated))
        self.assertEqual([True], hallucinated)

    def test_get_responses_extraction_predictions_empty_correct_3(self):
        raw_response = [{"raw_response":[]}]

        predictions, hallucinated  = get_responses_extraction_predictions(raw_response)
        print(predictions)

        self.assertEqual(1, len(hallucinated))
        self.assertEqual([False], hallucinated)

if __name__ == '__main__':
    unittest.main()
