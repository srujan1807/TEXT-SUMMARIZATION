#import pipeline from transformers
from transformers import pipeline

#define function that summarizes the text
def text_summarizer(text, max_length=150):
    #creating summarization pipeline
    # Explicitly use PyTorch implementation
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn", #bart model
        framework="pt"  # Force PyTorch usage
    )
    summary = summarizer(
        text,
        max_length=100, #maximum length for the summary 
        min_length=30,  #minimum length for the summary
        do_sample=False,
        truncation=True
    )
    return summary[0]['summary_text']

if __name__ == "__main__": #input text to summarize
    input_text = """
   Climate change refers to long-term alterations in temperature, precipitation, wind patterns, and 
   other elements of the Earth's climate system.These changes are largely driven by human activities,
   especially the burning of fossil fuels such as coal, oil, and gas. This leads to increased concentrations
   of greenhouse gases in the atmosphere, which trap heat and cause the planet to warm.
   The consequences include more frequent and severe weather events such as hurricanes, droughts, and wildfires,
   as well as rising sea levels and melting glaciers. Ecosystems and biodiversity are also at risk, and human health,
   food security, and water resources are expected to be increasingly affected. Efforts to combat climate change include
   international agreements like the Paris Accord, as well as national and local initiatives to reduce emissions and transition
   to renewable energy."""
    
    print("Original Text Length:", len(input_text))
    result = text_summarizer(input_text)
    print("\nSummary:", result)
    print("Summary Length:", len(result)) #print summary length 