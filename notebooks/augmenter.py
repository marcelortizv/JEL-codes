import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action


sentenceInserter = nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2')
abstractiveAugmenter = nas.AbstSummAug(model_path='t5-base')
synonymAugmenter = naw.SynonymAug(aug_src='wordnet')
wordSubstitute = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute")
wordInserter = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")


def _augment_text(text, label, n):
    augmented_texts = []

    augmented_texts.extend(sentenceInserter.augment(text, n=n//5))
    augmented_texts.extend(abstractiveAugmenter.augment(text, n=n//5))
    augmented_texts.extend(synonymAugmenter.augment(text, n=n//5))
    augmented_texts.extend(wordSubstitute.augment(text, n=n//5))
    augmented_texts.extend(wordInserter.augment(text, n=n//5))

    labels = [label]*n

    return augmented_texts, labels


def augment_text(texts, labels, n):

    org_n = len(texts)
    ns = n//org_n

    augmented_texts = []
    augmented_labels = []

    for text, label in zip(texts, labels):
        aug_text, aug_labels = _augment_text(text, label, ns)
        augmented_texts.extend(aug_text)
        augmented_labels.extend(aug_labels)
        
    return augmented_texts, augmented_labels

