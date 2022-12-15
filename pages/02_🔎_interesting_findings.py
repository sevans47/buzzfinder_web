import streamlit as st
import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def make_aug_eval_plot():

    # load results as a dataframe
    with open("data/aug_eval.json", 'r') as f:
        cv_results = json.load(f)

    cv_df = pd.DataFrame.from_dict(cv_results)

    # process dataframe
    cv_grouped = cv_df.groupby('type_of_augmentation').agg({
        'test_loss': ['mean', 'min', 'max'],
        'test_accuracy': ['mean', 'min', 'max']
    }).sort_values(by=('test_loss', 'mean'), ascending=False)

    # create function to auto highlight min / max bar
    def color_bars(column, min_max='min'):
        colors = ['gray'] * len(column)
        idx = column.argmin() if min_max == 'min' else column.argmax()
        colors[idx] = 'blue'
        return colors

    # make plot
    fig, axes = plt.subplots(3, 2, figsize=(10,10), sharey=True)
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle("Loss and Accuracy for types of Data Augmentation", fontsize=18, y=0.95)

    # reorganize column names
    means = [col for col in cv_grouped.columns if 'mean' in col]
    mins = [col for col in cv_grouped.columns if 'min' in col]
    maxes = [col for col in cv_grouped.columns if 'max' in col]
    cols = means + mins + maxes

    y = range(len(cv_grouped))
    alpha = 0.3

    # generate plots
    for col, ax in zip(cols, axes.ravel()):

        min_max = 'min' if 'loss' in col[0] else 'max'
        xlim = [0.05, 0.35] if 'loss' in col[0] else [0.9, 1.0]

        ax.barh(y, cv_grouped[col], color=color_bars(cv_grouped[col], min_max=min_max), alpha=alpha)
        ax.set_yticks(y)
        ax.set_yticklabels(cv_grouped.index)
        ax.set_xlim(xlim)
        ax.set_title(f"{col[1]} {col[0].split('_')[1]}")

    # show plot in streamlit app
    st.pyplot(fig)

def make_model_eval_plot():

    # dictionary of model results
    model_eval_dict = {
        'labels': ['highpass', 'all_aug_1to1', 'no_aug', 'all_aug_5to1'],
        'accuracy_values': [0.932, 0.936, 0.955, 0.958],
        'loss_values': [0.256, 0.265, 0.212, 0.218]
    }

    # function to change colors for chart
    def color_bars(values, min_max='min'):
        colors = ['gray'] * len(values)
        if min_max=='min':
            idx = min(range(len(values)), key=values.__getitem__)
        else:
            idx = max(range(len(values)), key=values.__getitem__)
        colors[idx] = 'blue'
        return colors

    # set up figure
    fig, axes = plt.subplots(1, 2, figsize=(10,3), sharey=True)
    plt.suptitle("Loss and Accuracy for models using different data augmentation", fontsize=18, y=1.1)

    y = range(len(model_eval_dict['labels']))
    alpha = 0.3
    xlim = [[0.2, 0.3], [0.9, 1.0]]
    min_max = ['min', 'max']

    # create plot
    for i, val in enumerate(['loss_values', 'accuracy_values']):

        axes[i].barh(y, model_eval_dict[val], color=color_bars(model_eval_dict[val], min_max[i]), alpha=alpha)
        axes[i].set_yticks(y)
        axes[i].set_yticklabels(model_eval_dict['labels'])
        axes[i].set_xlim(xlim[i])
        axes[i].set_title(val.replace("_", " "))

    # show plot in streamlit app
    st.pyplot(fig)

## page config
st.set_page_config(page_title="findings", page_icon="🔎",layout="wide")

st.title("Intersting Findings 🔎")

st.markdown("""
            Although there were many steps in developing the buzzfinder api, three steps in particular had interesting results:
            - Using cross validation to compare augmentation techniques
            - Evaluating models using differently augmented data
            - Testing the API with different microphones
""")

st.markdown("---")
st.header("Using cross validation to compare augmentation techniques")

st.write("""
        Data augmentation is the process of applying a transformation to original or authentic data in order to bolster
        the dataset.  For audio data, this includes techniques such as adding white noise to a clip, applying a
        high pass filter, and so on.  I wanted to augment my dataset due to its limited size, but I didn't know
        which augmentation techniques would be the most effective.  To solve this, I used my original audio clips
        and the Audiomentations python library to create a function to do do k-fold cross validation on the data set.
        The function does the following:
         """)
st.markdown("""
            1. split the audio clips into five equal groups
            2. create 1 augmented clip per original clip using a single augmentation technique, essentially doubling the number of audio clips
            3. take the first group (20% of the data) and assign it as the test set.
            4. take the other four groups and all their augmented counterparts and combine them into the training set.
            5. the augmented data for the test set is ignored.
            6. train a CNN model using the training and test set and record the loss and accuracy scores.
            7. repeat steps 3 through 6 for the remaining four groups
""")
st.markdown("""
            I ran this function using the following augmentation techniques from Audiomentations:
            - AddGaussianNoise
            - Gain
            - HighPassFilter (I did this one twice)
            - LowPassFilter
            - HighPassFilter and LowPassFilter
            - All four techniques
""")
st.write("Here are the results:")
make_aug_eval_plot()
st.markdown("""
            Key takeaways from the chart:
            - HighPassFilter and LowPassFilter helped improve the test results the most.  This makes sense as these can be used to help filter out low frequency / high frequency noise in a signal, resulting in a clearer signal for the model to learn from.
            - There is predictably some randomness in the results, demonstrated by the different scores for the two times I ran the test with HighPassFilter.
            - AddGaussianNoise returned the lowest test results.  Just like for humans, adding noise to the signal makes it harder for the CNN to classify the tone.
            - Combining data augmentation techniques tended to result in lower test results.  I was surprised that combining HighPassFilter and LowPassFilter made the model perform nearly as bad as AddGaussianNoise.
""")
st.markdown("""
            What I learned:
            - If the model will be used with noisy audio data, emphasize AddGaussianNoise.
            - Conversely, if the model will be used with clear audio data, emphasize HighPassFilter or LowPassFilter.
            - If the model will be used with audio data of various quality, combining augmentation techniques is the way to go.
""")

st.markdown("---")
st.header("Model evaluations with differently augmented data")

st.write("I created a model using each of the following datasets:")
st.markdown("""
            - no_aug: Original data only (ie no data augmentation)
            - highpass: Data augmentation using Audiomentations' HighPassFilter at a rate of one augmentation per audio clip
            - all_aug_1to1: Data augmentation using Audiomentations' HighPassFilter, LowPassFilter, Gain, and AddGaussianNoise at a rate of one augmentation per audio clip
            - all_aug_5to1: Same as above, but at a rate of five augmentations per audio clip
""")
st.write("Here's how the models performed:")
make_model_eval_plot()
no_aug = Image.open('data/lc_no_aug.png')
highpass = Image.open('data/lc_highpass_1to1.png')
all_aug_1to1 = Image.open('data/lc_all_aug_1to1.png')
all_aug_5to1 = Image.open('data/lc_all_aug_5to1.png')

st.subheader("Learning curves for each model")
st.write("**no_aug**")
st.image(no_aug)
st.write("**highpass**")
st.image(highpass)
st.write("**all_aug_1to1**")
st.image(all_aug_1to1)
st.write("**all_aug_5to1**")
st.image(all_aug_5to1)

st.markdown("""
            Key Takeaways:
            - All the models had very similar results, with the difference in accuracy being less than 3% between the lowest and highest scores.
            - When crossvalidating augmentation techniques (see the previous section), HighPassFilter had the best results.  It was interesting to see it score lower than using no augmented data.
            - No augmented data trained for more epochs with bumpier learning curves, and had a very good score.
            - I needed to up the amount of augmented data from 1to1 to 5to1 to see an improvement over no augmented data.  The learning curves for 5to1 were much smoother and steeper than for no_aug.
            - Although not shown here, no_aug had the fastest training time at 5.6 minutes, and 5to1 was slowest at 17.4 minutes
""")

st.markdown("""
            What I learned:
            - Data augmentation doesn't necessarily help the model get a higher evaluation score.  It all depends on the test set.
            - Rather, data augmentation should be used to make the model more robust against real world data.
            - More data = slower training, even if fewer epochs are needed.
""")
st.write("Speaking of real world data ... ")

st.markdown("---")
st.header("Testing the API with different microphones")

st.write("""
         Before deploying the API, I had one more test.  When I made the recordings for my dataset, I used my iPhone mic.
         But I imagined someone using the API at home, and using the built in microphone in their laptop to record their
         guitar playing.  So I did the same - I recorded several notes using my laptop mic, varying the tone, pitch,
         duration, and guitar type for each note.  I then did the same thing, but using my iPhone mic like before.
         I had 54 clips from my laptop mic, and 61 clips from my iPhone mic.
         I then used the four models from the previous section to predict the tones for all the recordings.
""")
st.write("""
         Surprising to me, all the models made the exact same predictions.  I ended up choosing the 5to1 model
         for my API, but I don't have any evidence that it would do any better with real world data than the other
         models, so I chose it based on having the highest accuracy score.
""")
st.markdown("""
         Here are the results of my experiment:
         - Laptop mic accuracy: 59.3% 😱
         - iPhone mic accuracy: 83.6% 😢
""")
st.markdown("""
        Some observations from the laptop mic experiment:
        - My model overpredicted clean.  44/54 (81.5%) clean predictions, but only 24/54 (44.4%) were actually clean.
        - My model did terrible predicting buzzy notes.  4/18 (22.2%) buzzy notes were correctly predicted.  The remaining 14 were all predicted as clean.
        - My model was bad at predicting muted notes.  5/12 (41.67) muted notes were correctly predicted.  6/7 reamining notes were predicted as clean.
        - The recordings were way quieter than with my iPhone mic, and had a lot more recording hiss.
""")
st.markdown("""
        Some observations from the iPhone mic experiment:
        - 21/25 (84.0%) clean notes were correctly predicted.
        - 18/24 (75.0%) buzzy notes were correctly predicted.
        - 12/12 (100%) muted notes were correctly predicted.
        - The model struggled a bit with high clean notes, thinking they were buzzy.
        - The model struggled a lot with low buzzy notes, thinking they were clean.
""")
st.markdown("""
        Based on these tests, here is how I would further improve the model:
        - add clips to training data using laptop's built-in mic with default settings (most likely will be the main way people use the API)
        - add clips using a variety of picking techniques
            - I didn't pay attention to using fingers vs. thumb, using nails vs. flesh, and using rest stroke vs. free stroke. Didn't use a pick at all.
            - I think this is a big reason why the model struggled with high clean notes and low buzzy notes - how you pick can add extra sound / noise, and I tended to really pop the notes. I should have played more smoothly
        - add clips using a variety of dynamics
            - overall I played really loud. Focusing on a variety of dynamics while minimizing attack sound would make it easier for the model to learn the different tones.
        - add clips using different types of acoustic guitars
            - This would be the final step once I'm able to get my current model able to generalize better with mic types, picking types, and dynamics. I could then make a list of types of clips I need and how many of each for adding new types of guitars to the model.

""")
