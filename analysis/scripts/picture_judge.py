import os
import pandas as pd
import streamlit as st

# Path configurations
base_dir = 'experiments/gardenpath_10_24/results/llm_results/drawings/'
models = os.listdir(base_dir)

# Load your dataframe
df = pd.read_csv('experiments/gardenpath_10_24/data/llm_data/drawing_experiment.csv')

# Load progress if available
try:
    progress_df = pd.read_csv('experiments/gardenpath_10_24/analysis/classification_progress.csv')
except FileNotFoundError:
    progress_df = pd.DataFrame(columns=['sent_id', 'classification', 'sent_type', 'sentence', 'model_name'])

# Function to get the next sentence to classify
def get_next_sentence(curr_mod):
    curr_df = progress_df[progress_df.model_name == curr_mod]
    classified_ids = set(curr_df['sent_id'])
    for _, row in df.iterrows():
        if row['sent_id'] not in classified_ids:
            return row['sent_type'], row['sent_id'], row['sentence']
    return None, None, None

for model_name in models[3:4]:

    st.title("Image Classification Tool")

    # Retrieve the next sentence and image data to classify
    sent_type, sent_id, sentence = get_next_sentence(model_name)

    if sent_id is not None:
        # Construct image path
        image_path = os.path.join(base_dir, model_name, f'{sent_id}.png')

        # Check if image exists
        if os.path.exists(image_path):
            st.header(f"{sentence}")

            # Display image
            st.image(image_path, caption=f'Sentence Type: {sent_type}', use_column_width=True)

            # Radio button selection for classification
            category = st.radio("Classify the image", 
                                ["correctly understood", 
                                 "partial misunderstanding", 
                                 "complete misunderstanding", 
                                 "not applicable"], 
                                index=0)

            if st.button('Submit Classification'):
                # Update progress dataframe and save it
                progress_df.loc[len(progress_df)] = [sent_id, category, sent_type, sentence, model_name]
                progress_df.to_csv('experiments/gardenpath_10_24/analysis/classification_progress.csv', index=False)
                st.success("Classification submitted!")
                st.rerun()  # Reload the page for the next image

        else:
            st.warning("No image available for this sentence.")
            if st.button('Mark as No Image'):
                progress_df.loc[len(progress_df)] = [sent_id, 'no image', sent_type, sentence, model_name]
                progress_df.to_csv('experiments/gardenpath_10_24/analysis/classification_progress.csv', index=False)
                st.success("Marked as no image!")
                st.rerun()


    else:
        st.info(f"All sentences classified for model {model_name}. Thank you!")
        st.rerun()
            