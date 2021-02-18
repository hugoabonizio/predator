import torch
from predator import Predator
import pandas as pd


def test_sanity_check():
    data = {
        "text": [
            "How far is it from Denver to Aspen?",
            "What county is Modesto, California in?",
            "Who was Galileo?",
            "What is an atom?",
            "When did Hawaii become a state?",
            "How tall is the Sears Building?",
            "George Bush purchased a small interest in which baseball team?",
            "What is Australia's national flower?",
            "Why does the moon turn orange?",
            "What is autism?",
            "What city had a world fair in 1900?",
            "What person's head is on a dime?",
            "What is the average weight of a Yellow Labrador?",
            "Who was the first man to fly across the Pacific Ocean?",
            "When did Idaho become a state?",
            "What is the life expectancy for crickets?",
            "What metal has the highest melting point?",
            "Who developed the vaccination against polio?",
            "What is epilepsy?",
            "What year did the Titanic sink?",
            "Who was the first American to walk in space?",
            "What is a biosphere?",
            "What river in the US is known as the Big Muddy?",
            "What is bipolar disorder?",
            "What is cholesterol?",
            "Who developed the Macintosh computer?",
            "What is caffeine?",
            "What imaginary line is halfway between the North and South Poles?",
            "Where is John Wayne airport?",
            "What hemisphere is the Philippines in?",
            "What is the average speed of the horses at the Kentucky Derby?",
        ],
        "label": [0] * 15 + [1] * 16,
    }
    df_train = df_valid = pd.DataFrame(data)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(df_train)
    print(df_valid)

    predator = Predator(df_train, df_valid, device=device)
    predator.train()

    df_aug = predator.augment(augment_ratio=2.0)
    print(df_aug)
    assert df_aug is not None
