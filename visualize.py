from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline
plt.ion()   # interactive mode
import seaborn as sns


def see_samples(train_df):
    fig = plt.figure(figsize=(15, 15))
    columns, rows = 3, 2
    end, start = train_df.loc[0:].shape
    ax = []
    import random
    for i in range(columns*rows):
        # img = np.array(Image.open(train_img_path.values[k][0]))
        k = random.randint(start, end)
        img = mpimg.imread((train_df.iloc[k,0]))
        title = (train_df.iloc[k,0]).split('/')
        title = title[2]+'-'+title[3]+'-'+title[4]
        k += 1
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title(title)  # set title
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap="gray")
    plt.tight_layout(True)
    plt.show()  # finally, render the plot



def view_data_count(train_df, valid_df):
    sns.countplot(train_df['Label'])
    plt.figure(figsize=(15,7))
    sns.countplot(data=train_df,x='BodyPart',hue='Label')
    plt.figure(figsize=(15,7))
    sns.countplot(data=train_df,x='StudyType',hue='Label')

    sns.countplot(valid_df['Label'])
    plt.figure(figsize=(15,7))
    sns.countplot(data=valid_df,x='BodyPart',hue='Label')
    plt.figure(figsize=(15,7))
    sns.countplot(data=valid_df,x='StudyType',hue='Label')
