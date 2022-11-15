# Music_Recommender_System

## Dataset:
<ul>
<li>The million song dataset is a very popular dataset and is available at [Echnonest Taste Profile Subset](http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/train_triplets.txt.zip)<br>
<li>The dataset ws created as a collaborative project between the Echonest and LABRosa. The dataset we have is only a subset of the million songs dataset.<br>
<li>The dataset contains 48 million lines of triplets. Each triplet contains (user id, song id, play counts).
<li>The overall dataset contains around a million unique users and around 384,000 songs from the million song dataset contained in it.
<li>Upon decompression, the txt file takes a size of 2.79 GB<br>
</ul>

## Exploratory Data Analysis (EDA):
EDA is important for this dataset as it is large. It will lead us to information that we can use to trim down the dataset a little.
### Loading and Trimming Data:
We will only load about 10000 rows from the entire dataset to see what it looks like.<br>

```
triplet_dataset = pd.read_csv(filepath_or_buffer = filepath,nrows = 10000,sep = '\t',names = ['user','song','play_count'])
```
Now, let's determine how many users account for maximum percentage of play counts so the focus of analysis is on those users<br>
```
#Determine how many unique users does the dataset have. 
#So we concentrate on users that consitute to a large percentage of play counts
#Due to the large size of the file, we will read it line by line
#We will then extract play count information on a user(or song)


output_dict = {}

with open(filepath) as f:
    for line_number,line in enumerate(f):
        user = line.split('\t')[0]
        play_count = int(line.split('\t')[2])
        if user in output_dict:
            play_count += output_dict[user]
            output_dict.update({user:play_count})
        output_dict.update({user:play_count})
    output_list = [{'user':k,'play_count':v} for k,v in output_dict.items()]
    play_count_df = pd.DataFrame(output_list)
    play_count_df = play_count_df.sort_values(by='play_count',ascending = False)
    play_count_df.to_csv(path_or_buf = '../data/user_playcount_df.csv',index = False)
```

