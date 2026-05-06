from jiwer import cer, wer

# Your actual samples from the output
refs = [
    'assuredness " Bella Bella Marie " ( Parlophone ) , a lively song that changes tempo mid-way .',
    "I don't think he will storm the charts with this one , but it 's a good start .",
    'CHRIS CHARLES , 39 , who lives in Stockton-on-Tees , is an accountant .',
    "Become a success with a disc and hey presto ! You 're a star ... . Rolly sings with",
    'Tolch , as he is known in Tin Pan Alley , likes songs with a month in the title . He wrote'
]

preds = [
    'assuredness " Bella Bella Marie " ( Parlophone ) , a lively song that changes tempo mid-way .',
    "I don't think he will storm the charts with this one , but it 's a good start .",
    'CHRIS CHARLES , 39 , who lives in Stocuton-on-Tees , is an accountant .',
    "Become a success with a dice and hey presto ! You 're a star ... . Rolly sings with",
    'Tolch , as he is known in Tin Pan Alley , writes songs with a month in the title . He wrote'
]

print(f"Manual Test (5 samples):")
print(f"CER: {cer(refs, preds):.4f}")
print(f"WER: {wer(refs, preds):.4f}")
