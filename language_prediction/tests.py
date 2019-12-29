import os

from language_prediction.dataset_readers import TextExpDataSetReader
from language_prediction.models import BasicTextModel

from allennlp.common.testing import AllenNlpTestCase, ModelTestCase
from allennlp.common.util import ensure_list
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.modules.attention.dot_product_attention import DotProductAttention
from allennlp.modules.feedforward import FeedForward
from allennlp.training.trainer import Trainer
import torch.optim as optim

base_directory = os.path.abspath(os.curdir)
data_directory = os.path.join(base_directory, 'data')


class TestTextExpDataSetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        # the token indexer is responsible for mapping tokens to integers
        token_indexer = ELMoTokenCharactersIndexer()

        def tokenizer(x: str):
            return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words(x)]

        reader = TextExpDataSetReader(token_indexers=token_indexer, tokenizer=tokenizer)
        instances = ensure_list(reader.read(os.path.join(data_directory, 'test_code_data.csv')))
        # TODO: add the numbers to the test
        instance0 = {
            'sequence_review': [
                ['Positive', ':', 'Extremely', 'helpful', 'and', 'friendly', 'staff', 'hotel', 'in', 'great', 'shape', 'and', 'location', '.', 'Would', 'def', 'reccomend', 'and', 'go', 'back', 'again', '.', 'They', 'deserve', 'all', 'the', 'credit', 'they', 'get', '.', 'Negative', ':', 'Not', 'a', 'single', 'thing', '.'],
                ['Positive', ':', 'Location', '.', 'Location', '.', 'Location', '.', 'Room', 'small', 'but', 'perfectly', 'formed', '.', 'Staff', 'very', 'helpful', 'and', 'accommodated', 'a', 'change', 'to', 'the', 'offered', 'menu', '.', 'Decor', 'modern', 'and', 'tasteful', '.', 'Negative', ':', '.'],
                ['Positive', ':', 'Pool', 'was', 'great', 'with', 'amazing', 'views', 'cocktails', 'on', 'the', 'roof', 'at', 'night', 'were', 'perfect', '.', 'Good', 'wifi', ',', 'location', 'to', 'the', 'metro', 'was', 'excellent', 'not', 'so', 'many', 'bars', 'restaurants', 'nearby', 'but', 'easy', 'to', 'travel', 'into', 'central', 'Barcelona', '.', 'Room', 'was', 'spacious', '.', 'Staff', 'helpful', 'and', 'barman', 'was', 'fab', 'made', 'us', 'cocktails', 'not', 'on', 'the', 'menu', '.', 'Very', 'clean', 'everywhere', '.', 'Will', 'definitely', 'be', 'back', 'to', 'Barcelona', 'and', 'would', 'stay', 'here', 'again', '.', 'Negative', ':', 'No', 'tea', 'coffee', 'making', 'facilities', 'in', 'the', 'room', 'but', 'we', 'knew', 'that', 'when', 'we', 'booked', '.', 'I', "'m", 'a', 'typical', 'Brit', 'who', 'likes', 'her', 'Tea', '.', 'Breakfast', 'was', 'slightly', 'overpriced', 'but', 'you', 'did', 'have', 'a', 'fantastic', 'selection', '.'],
                ['Negative', ':', 'You', 'need', 'a', 'car', 'if', 'you', 'want', 'to', 'stay', 'at', 'this', 'hotel', 'with', 'the', 'family', '.', 'Parking', 'is', 'around', '10', 'euros', 'per', 'day', 'but', 'you', 'can', 'park', 'somewhere', 'on', 'the', 'street', 'near', 'the', 'hotel', 'for', 'free', '.', 'There', 'are', 'no', 'other', 'facilities', 'in', 'the', 'hotel', 'beside', 'the', 'gym', 'which', 'is', 'free', 'and', 'the', 'spa', 'that', 'costs', '50', 'euros', 'per', 'hour', 'max', '6', 'persons', 'which', 'is', 'still', 'expensive', 'for', 'a', 'family', '.', 'Positive', ':', 'The', 'bed', 'was', 'very', 'comfortable', ',', 'the', 'room', 'and', 'the', 'bathroom', 'were', 'clean', 'and', 'nicely', 'designed', '.'],
                ['Negative', ':', 'The', 'entrance', 'is', 'inconspicuous', 'one', 'could', 'say', 'hidden', 'not', 'so', 'easy', 'to', 'spot', 'but', 'security', 'is', 'good', '.', 'Just', 'do', 'not', 'expect', 'a', 'big', 'reception', 'it', "'s", 'on', 'the', 'first', 'floor', '.', 'Positive', ':', 'Largest', 'room', 'we', 'ever', 'had', 'in', 'Paris', '.', 'Excellent', 'breakfast', '.', 'Very', 'convenient', 'location', 'for', 'us', 'in', 'front', 'of', 'Gare', 'de', 'l', 'Est', 'and', 'walking', 'distance', 'to', 'Gare', 'du', 'Nord', '.'],
                ['Negative', ':', 'everything', 'how', 'such', 'a', 'facility', 'can', 'take', '4', 'stars', '.', 'The', 'room', 'was', 'dirty', '.', 'Even', 'in', 'bathroom', 'there', 'were', 'hair', 'and', 'dirty', 'everywhere', '.', 'Very', 'small', 'uncomfortable', '.', 'Positive', ':', 'nothing', 'except', 'location', '.'],
                ['Negative', ':', 'The', 'hotel', 'buffet', 'was', 'okay', 'but', 'not', 'great', 'and', 'discovered', 'that', 'I', 'had', 'been', 'overcharged', 'after', 'I', 'had', 'checked', 'out', ',', 'charged', 'for', '4', 'adults', 'instead', 'of', 'one', 'adult', 'and', 'three', 'children', 'as', 'on', 'original', 'bill', '.', 'Positive', ':', 'Room', 'was', 'very', 'comfortable', 'and', 'clean', 'with', 'a', '/', 'c', 'and', 'a', 'small', 'fridge', 'and', 'kettle', 'provided', '.', 'Excellent', 'location', 'and', 'great', 'view', 'of', 'the', 'Seine', 'from', 'our', 'room', ',', 'would', 'definitely', 'love', 'to', 'stay', 'in', 'this', 'hotel', 'again', '.'],
                ['Negative', ':', 'I', 'felt', 'some', 'elements', 'of', 'breakfast', 'could', 'have', 'been', 'better', '.', 'For', 'example', 'the', 'tea', 'we', 'were', 'served', 'was', 'only', 'luke', 'warm', 'and', 'the', 'buffet', 'was', 'not', 'always', 'fully', 'topped', 'up', '.', 'Positive', ':', 'Staff', 'very', 'welcoming', 'and', 'friendly', 'and', 'the', 'hotel', 'and', 'room', 'lovely', '.'],
                ['Negative', ':', 'The', 'Location', 'of', 'the', 'room', 'next', 'to', 'the', 'Elevator', 'was', 'not', 'the', 'key', 'but', 'the', 'slamming', 'emergency', 'door', 'which', 'was', 'used', 'many', 'times', 'for', 'what', 'reason', 'ever', '.', 'The', 'gap', 'between', 'door', 'and', 'floor', 'let', 'my', 'room', 'lightening', 'up', 'like', 'having', 'forgotten', 'to', 'Switch', 'off', 'the', 'lamp', '.', 'Positive', ':', 'Friendly', 'staff', 'especially', 'in', 'the', 'welcome', 'area', '.', 'Location', 'of', 'the', 'Hotel', 'in', 'the', 'middle', 'of', 'many', 'famous', 'streets', '.'],
                ['Negative', ':', 'The', 'showers', 'looked', 'modern', 'however', 'the', 'pressure', 'of', 'water', 'coming', 'out', 'of', 'the', 'shower', 'head', 'was', 'average', 'at', 'best', '.', 'Positive', ':', 'The', 'interior', 'was', 'sleek', 'and', 'relatively', 'modern', 'which', 'was', 'surprising', 'giving', 'that', 'the', 'exterior', 'of', 'the', 'hotel', 'was', "n't", 'on', 'par', '.']],
            'label': 4,
            'metadata': {'k_size': 10, 'pair_id': '91ol4nv6_4', 'sample_id': '91ol4nv6_4_10'}
        }

        instance4 = {
            'sequence_review': [
                ['Positive', ':', 'Extremely', 'helpful', 'and', 'friendly', 'staff', 'hotel', 'in', 'great', 'shape', 'and', 'location', '.', 'Would', 'def', 'reccomend', 'and', 'go', 'back', 'again', '.', 'They', 'deserve', 'all', 'the', 'credit', 'they', 'get', '.', 'Negative', ':', 'Not', 'a', 'single', 'thing', '.'],
                ['Positive', ':', 'Location', '.', 'Location', '.', 'Location', '.', 'Room', 'small', 'but', 'perfectly', 'formed', '.', 'Staff', 'very', 'helpful', 'and', 'accommodated', 'a', 'change', 'to', 'the', 'offered', 'menu', '.', 'Decor', 'modern', 'and', 'tasteful', '.', 'Negative', ':', '.'],
                ['Positive', ':', 'Pool', 'was', 'great', 'with', 'amazing', 'views', 'cocktails', 'on', 'the', 'roof', 'at', 'night', 'were', 'perfect', '.', 'Good', 'wifi', ',', 'location', 'to', 'the', 'metro', 'was', 'excellent', 'not', 'so', 'many', 'bars', 'restaurants', 'nearby', 'but', 'easy', 'to', 'travel', 'into', 'central', 'Barcelona', '.', 'Room', 'was', 'spacious', '.', 'Staff', 'helpful', 'and', 'barman', 'was', 'fab', 'made', 'us', 'cocktails', 'not', 'on', 'the', 'menu', '.', 'Very', 'clean', 'everywhere', '.', 'Will', 'definitely', 'be', 'back', 'to', 'Barcelona', 'and', 'would', 'stay', 'here', 'again', '.', 'Negative', ':', 'No', 'tea', 'coffee', 'making', 'facilities', 'in', 'the', 'room', 'but', 'we', 'knew', 'that', 'when', 'we', 'booked', '.', 'I', "'m", 'a', 'typical', 'Brit', 'who', 'likes', 'her', 'Tea', '.', 'Breakfast', 'was', 'slightly', 'overpriced', 'but', 'you', 'did', 'have', 'a', 'fantastic', 'selection', '.'],
                ['Negative', ':', 'You', 'need', 'a', 'car', 'if', 'you', 'want', 'to', 'stay', 'at', 'this', 'hotel', 'with', 'the', 'family', '.', 'Parking', 'is', 'around', '10', 'euros', 'per', 'day', 'but', 'you', 'can', 'park', 'somewhere', 'on', 'the', 'street', 'near', 'the', 'hotel', 'for', 'free', '.', 'There', 'are', 'no', 'other', 'facilities', 'in', 'the', 'hotel', 'beside', 'the', 'gym', 'which', 'is', 'free', 'and', 'the', 'spa', 'that', 'costs', '50', 'euros', 'per', 'hour', 'max', '6', 'persons', 'which', 'is', 'still', 'expensive', 'for', 'a', 'family', '.', 'Positive', ':', 'The', 'bed', 'was', 'very', 'comfortable', ',', 'the', 'room', 'and', 'the', 'bathroom', 'were', 'clean', 'and', 'nicely', 'designed', '.'],
                ['Negative', ':', 'The', 'entrance', 'is', 'inconspicuous', 'one', 'could', 'say', 'hidden', 'not', 'so', 'easy', 'to', 'spot', 'but', 'security', 'is', 'good', '.', 'Just', 'do', 'not', 'expect', 'a', 'big', 'reception', 'it', "'s", 'on', 'the', 'first', 'floor', '.', 'Positive', ':', 'Largest', 'room', 'we', 'ever', 'had', 'in', 'Paris', '.', 'Excellent', 'breakfast', '.', 'Very', 'convenient', 'location', 'for', 'us', 'in', 'front', 'of', 'Gare', 'de', 'l', 'Est', 'and', 'walking', 'distance', 'to', 'Gare', 'du', 'Nord', '.'],
                ['Negative', ':', 'everything', 'how', 'such', 'a', 'facility', 'can', 'take', '4', 'stars', '.', 'The', 'room', 'was', 'dirty', '.', 'Even', 'in', 'bathroom', 'there', 'were', 'hair', 'and', 'dirty', 'everywhere', '.', 'Very', 'small', 'uncomfortable', '.', 'Positive', ':', 'nothing', 'except', 'location', '.']],
            'label': 4,
            'metadata': {'k_size': 6, 'pair_id': '91ol4nv6_4', 'sample_id': '91ol4nv6_4_6'}
        }

        instance12 = {
            'sequence_review': [
                ['Positive', ':', 'Largest', 'room', 'we', 'ever', 'had', 'in', 'Paris', '.', 'Excellent', 'breakfast', '.', 'Very', 'convenient', 'location', 'for', 'us', 'in', 'front', 'of', 'Gare', 'de', 'l', 'Est', 'and', 'walking', 'distance', 'to', 'Gare', 'du', 'Nord', '.', 'Negative', ':', 'The', 'entrance', 'is', 'inconspicuous', 'one', 'could', 'say', 'hidden', 'not', 'so', 'easy', 'to', 'spot', 'but', 'security', 'is', 'good', '.', 'Just', 'do', 'not', 'expect', 'a', 'big', 'reception', 'it', "'s", 'on', 'the', 'first', 'floor', '.'],
                ['Positive', ':', 'Excellent', 'breakfast', 'and', 'friendly', 'helpful', 'staff', '.', 'Good', 'location', 'close', 'to', 'the', 'Metro', 'station', 'and', 'walking', 'distance', 'to', 'Sagrada', 'Familia', '.', 'Nice', 'snack', 'bar', 'area', 'to', 'grab', 'a', 'light', 'meal', '.', 'We', 'would', 'stay', 'there', 'again', '.', 'Negative', ':', 'Tried', 'to', 'visit', 'the', 'Fitness', 'centre', 'Spa', 'at', '5:00', 'in', 'the', 'evening', 'but', 'it', 'was', 'closed', '.', 'Did', "n't", 'get', 'to', 'see', 'it', 'so', 'I', 'ca', "n't", 'comment', '.'],
                ['Negative', ':', 'Rooms', 'were', 'tired', '.', 'Carpet', 'needed', 'a', 'good', 'clean', 'or', 'replacement', '.', 'Plumbing', 'system', 'outdated', '.', 'Various', 'fittings', 'were', 'missing', 'or', 'knobs', 'had', 'come', 'off', '.', 'There', 'were', 'about', '5', 'lamps', 'that', 'needed', 'replacing', '.', 'The', 'tv', 'remote', 'did', 'not', 'work', 'and', 'request', 'for', 'new', 'batteries', 'did', 'not', 'happen', '.', 'Unfortunately', '2', 'of', 'our', 'party', 'were', 'ill', 'and', 'stayed', 'in', 'their', 'rooms', 'and', 'were', 'disturbed', 'even', 'though', 'we', 'had', 'requested', 'that', 'the', 'rooms', 'were', 'not', 'serviced', 'that', 'day', '.', 'Nothing', 'to', 'do', 'with', 'the', 'hotel', 'but', 'there', 'is', 'a', '10', 'euro', 'per', 'night', 'city', 'tax', '.', 'Positive', ':', 'Trams', 'passed', 'the', 'front', 'door', '.', 'Attractive', 'foyer', '.', 'Staff', 'spoke', 'English', 'and', 'gave', 'good', 'guidance', 'to', 'how', 'to', 'get', 'around', '.', 'Breakfast', 'waitress', 'was', 'very', 'helpful', 'but', 'the', 'selection', 'of', 'food', 'was', 'limited', 'and', 'breakfast', 'finished', 'at', '09:00', '.'],
                ['Negative', ':', 'The', 'showers', 'looked', 'modern', 'however', 'the', 'pressure', 'of', 'water', 'coming', 'out', 'of', 'the', 'shower', 'head', 'was', 'average', 'at', 'best', '.', 'Positive', ':', 'The', 'interior', 'was', 'sleek', 'and', 'relatively', 'modern', 'which', 'was', 'surprising', 'giving', 'that', 'the', 'exterior', 'of', 'the', 'hotel', 'was', "n't", 'on', 'par', '.'],
                ['Positive', ':', 'Great', 'hotel', '.', 'Friendly', 'and', 'very', 'helpful', 'staff', '.', 'Spotless', '.', 'Negative', ':', 'Booked', 'a', 'double', 'room', '.', 'Surprised', 'and', 'disappointed', 'that', 'this', 'was', 'infact', 'two', 'single', 'beds', 'joined', 'together', '.'],
                ['Positive', ':', 'Everything', 'was', 'perfect', '.', 'They', 'also', 'upgraded', 'us', 'as', 'a', 'surprise', 'for', 'my', 'husbands', 'birthday', 'we', 'had', 'the', 'most', 'awesome', 'views', 'and', 'the', 'room', 'was', 'perfect', ',', 'we', 'woke', 'up', 'to', 'see', 'the', 'sunrise', '.', 'our', 'stay', 'was', 'simply', 'perfect', 'and', 'very', 'recommended', '.', 'I', 'recently', 'stayed', 'at', 'the', 'Mondrian', 'hotel', 'priced', 'almost', 'the', 'same', 'for', 'the', 'room', 'categories', 'but', 'in', 'terms', 'of', 'service', 'experience', 'attention', 'to', 'detail', 'and', 'customer', 'satisfaction', 'this', 'hotel', 'by', 'FAR', 'exceeded', 'that', 'experience', 'so', 'much', 'so', 'we', 'joined', 'up', 'to', 'Shangri', 'la', "'s", 'loyalty', 'program', 'as', 'I', 'was', 'really', 'surprised', 'we', 'could', 'still', 'get', 'such', 'amazing', 'customer', 'service', '.', 'Fully', 'recommend', 'staying', 'here', 'did', 'I', 'mention', 'the', 'phenomenal', 'views', '.', 'Negative', ':', 'Nothing', 'not', 'to', 'like', '.'],
                ['Negative', ':', 'The', 'hotel', 'buffet', 'was', 'okay', 'but', 'not', 'great', 'and', 'discovered', 'that', 'I', 'had', 'been', 'overcharged', 'after', 'I', 'had', 'checked', 'out', ',', 'charged', 'for', '4', 'adults', 'instead', 'of', 'one', 'adult', 'and', 'three', 'children', 'as', 'on', 'original', 'bill', '.', 'Positive', ':', 'Room', 'was', 'very', 'comfortable', 'and', 'clean', 'with', 'a', '/', 'c', 'and', 'a', 'small', 'fridge', 'and', 'kettle', 'provided', '.', 'Excellent', 'location', 'and', 'great', 'view', 'of', 'the', 'Seine', 'from', 'our', 'room', ',', 'would', 'definitely', 'love', 'to', 'stay', 'in', 'this', 'hotel', 'again', '.'],
                ['Positive', ':', 'Excellent', 'hotel', 'at', 'the', 'city', 'center', '.', 'Hotel', 'is', 'very', 'new', 'and', 'modern', '.', 'Staff', 'is', 'professional', 'and', 'helpful', '.', 'Location', 'is', 'perfect', 'at', 'the', 'city', 'center', '.', 'Negative', ':', '.']],
            'label': 2,
            'metadata': {'k_size': 8, 'pair_id': 'd9oijkzb_12', 'sample_id': 'd9oijkzb_12_8'}
        }

        instance15 = {
            'sequence_review': [
                ['Positive', ':', 'Largest', 'room', 'we', 'ever', 'had', 'in', 'Paris', '.', 'Excellent', 'breakfast',
                 '.', 'Very', 'convenient', 'location', 'for', 'us', 'in', 'front', 'of', 'Gare', 'de', 'l', 'Est',
                 'and', 'walking', 'distance', 'to', 'Gare', 'du', 'Nord', '.', 'Negative', ':', 'The', 'entrance',
                 'is', 'inconspicuous', 'one', 'could', 'say', 'hidden', 'not', 'so', 'easy', 'to', 'spot', 'but',
                 'security', 'is', 'good', '.', 'Just', 'do', 'not', 'expect', 'a', 'big', 'reception', 'it', "'s",
                 'on', 'the', 'first', 'floor', '.'],
                ['Positive', ':', 'Excellent', 'breakfast', 'and', 'friendly', 'helpful', 'staff', '.', 'Good',
                 'location', 'close', 'to', 'the', 'Metro', 'station', 'and', 'walking', 'distance', 'to', 'Sagrada',
                 'Familia', '.', 'Nice', 'snack', 'bar', 'area', 'to', 'grab', 'a', 'light', 'meal', '.', 'We', 'would',
                 'stay', 'there', 'again', '.', 'Negative', ':', 'Tried', 'to', 'visit', 'the', 'Fitness', 'centre',
                 'Spa', 'at', '5:00', 'in', 'the', 'evening', 'but', 'it', 'was', 'closed', '.', 'Did', "n't", 'get',
                 'to', 'see', 'it', 'so', 'I', 'ca', "n't", 'comment', '.'],
                ['Negative', ':', 'Rooms', 'were', 'tired', '.', 'Carpet', 'needed', 'a', 'good', 'clean', 'or',
                 'replacement', '.', 'Plumbing', 'system', 'outdated', '.', 'Various', 'fittings', 'were', 'missing',
                 'or', 'knobs', 'had', 'come', 'off', '.', 'There', 'were', 'about', '5', 'lamps', 'that', 'needed',
                 'replacing', '.', 'The', 'tv', 'remote', 'did', 'not', 'work', 'and', 'request', 'for', 'new',
                 'batteries', 'did', 'not', 'happen', '.', 'Unfortunately', '2', 'of', 'our', 'party', 'were', 'ill',
                 'and', 'stayed', 'in', 'their', 'rooms', 'and', 'were', 'disturbed', 'even', 'though', 'we', 'had',
                 'requested', 'that', 'the', 'rooms', 'were', 'not', 'serviced', 'that', 'day', '.', 'Nothing', 'to',
                 'do', 'with', 'the', 'hotel', 'but', 'there', 'is', 'a', '10', 'euro', 'per', 'night', 'city', 'tax',
                 '.', 'Positive', ':', 'Trams', 'passed', 'the', 'front', 'door', '.', 'Attractive', 'foyer', '.',
                 'Staff', 'spoke', 'English', 'and', 'gave', 'good', 'guidance', 'to', 'how', 'to', 'get', 'around',
                 '.', 'Breakfast', 'waitress', 'was', 'very', 'helpful', 'but', 'the', 'selection', 'of', 'food', 'was',
                 'limited', 'and', 'breakfast', 'finished', 'at', '09:00', '.'],
                ['Negative', ':', 'The', 'showers', 'looked', 'modern', 'however', 'the', 'pressure', 'of', 'water',
                 'coming', 'out', 'of', 'the', 'shower', 'head', 'was', 'average', 'at', 'best', '.', 'Positive', ':',
                 'The', 'interior', 'was', 'sleek', 'and', 'relatively', 'modern', 'which', 'was', 'surprising',
                 'giving', 'that', 'the', 'exterior', 'of', 'the', 'hotel', 'was', "n't", 'on', 'par', '.'],
                ['Positive', ':', 'Great', 'hotel', '.', 'Friendly', 'and', 'very', 'helpful', 'staff', '.', 'Spotless',
                 '.', 'Negative', ':', 'Booked', 'a', 'double', 'room', '.', 'Surprised', 'and', 'disappointed', 'that',
                 'this', 'was', 'infact', 'two', 'single', 'beds', 'joined', 'together', '.']],
            'label': 2,
            'metadata': {'k_size': 5, 'pair_id': 'd9oijkzb_12', 'sample_id': 'd9oijkzb_12_5'}
        }

        instance19 = {
            'sequence_review': [
                ['Positive', ':', 'Largest', 'room', 'we', 'ever', 'had', 'in', 'Paris', '.', 'Excellent', 'breakfast',
                 '.', 'Very', 'convenient', 'location', 'for', 'us', 'in', 'front', 'of', 'Gare', 'de', 'l', 'Est',
                 'and', 'walking', 'distance', 'to', 'Gare', 'du', 'Nord', '.', 'Negative', ':', 'The', 'entrance',
                 'is', 'inconspicuous', 'one', 'could', 'say', 'hidden', 'not', 'so', 'easy', 'to', 'spot', 'but',
                 'security', 'is', 'good', '.', 'Just', 'do', 'not', 'expect', 'a', 'big', 'reception', 'it', "'s",
                 'on', 'the', 'first', 'floor', '.']],
            'label': 2,
            'metadata': {'k_size': 1, 'pair_id': 'd9oijkzb_12', 'sample_id': 'd9oijkzb_12_1'}
        }

        # tests:
        # test sizes:
        # number of instances
        assert len(instances) == 20
        # length of sequence_review:
        seq_lengths = {
            0: 10, 1: 9, 2: 8, 3: 7, 4: 6, 5: 5, 6: 4, 7: 3, 8: 2, 9: 1,
            10: 10, 11: 9, 12: 8, 13: 7, 14: 6, 15: 5, 16: 4, 17: 3, 18: 2, 19: 1
        }
        for row, seq_length in seq_lengths.items():
            assert len(instances[row].fields['sequence_review'].field_list) == seq_length

        # same pair_id with the same label and the K_size compatible with the sequence_review length
        for instance_index in range(len(instances)):
            assert (instances[instance_index].fields['metadata'].metadata['pair_id'] == '91ol4nv6_4' and
                    instances[instance_index].fields['label'].label == 4) or\
                   (instances[instance_index].fields['metadata'].metadata['pair_id'] == 'd9oijkzb_12' and
                    instances[instance_index].fields['label'].label == 2)
            assert len(instances[instance_index].fields['sequence_review']) ==\
                   instances[instance_index].fields['metadata'].metadata['k_size']

        # compare specific instances
        for instance_num, instance in [[0, instance0], [4, instance4], [12, instance12], [15, instance15],
                                       [19, instance19]]:
            fields = instances[instance_num].fields
            assert [[t.text for t in fields['sequence_review'][i].tokens] for i in
                    range(len(fields['sequence_review'].field_list))] == instance['sequence_review']
            assert fields['label'].label == instance['label']
            assert fields['metadata'].metadata == instance['metadata']


class ExperimentClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/academic_paper_classifier.json',
                          'tests/fixtures/s2_papers.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


def manually_test_reader():
    token_indexer = ELMoTokenCharactersIndexer()

    def tokenizer(x: str):
        return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words(x)]

    reader = TextExpDataSetReader(token_indexers=token_indexer, tokenizer=tokenizer)
    instances = reader.read(os.path.join(data_directory, 'test_code_data.csv'))


def main():
    test_text_exp_data_set_reader_obj = TestTextExpDataSetReader()
    test_text_exp_data_set_reader_obj.test_read_from_file()
    experiment_classifier_test_obj = ExperimentClassifierTest()
    experiment_classifier_test_obj.test_model_can_train_save_and_load()


if __name__ == '__main__':
    main()
