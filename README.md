# Music_Recommender_System

## Dataset:
<ul>
<li>The million song dataset is a very popular dataset and is available at 
    <a href = "http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/train_triplets.txt.zip">Echnonest Taste Profile Subset</a><br>
<li>The dataset ws created as a collaborative project between the Echonest and LABRosa. The dataset we have is only a subset of the million songs dataset.<br>
<li>The dataset contains 48 million lines of triplets. Each triplet contains (user id, song id, play counts).
<li>The overall dataset contains around a million unique users and around 384,000 songs from the million song dataset contained in it.
<li>Upon decompression, the txt file takes a size of 2.79 GB<br>
<li> Place your train_triplets.txt file in the data folder to execute this code.<br>
</ul>

## Exploratory Data Analysis (EDA):
EDA is important for this dataset as it is large. It will lead us to information that we can use to trim down the dataset a little.
### Loading and Trimming Data:
Let's load the dataset to see what it looks like.<br>

```
filepath = '../data/train_triplets.txt' 
triplet_dataset = pd.read_csv(filepath_or_buffer = filepath,header = None,sep = '\t',names = ['user','song','play_count'])
```
#### Sample rows of the dataset:<br>
![image](https://user-images.githubusercontent.com/105756607/202028426-52b3dced-254a-4fd0-88ba-0becc0ddbb84.png)


The dataset has about 48373586 rows and 3 columns<br>

Now, let's determine how many users account for maximum percentage of play counts so the focus of analysis is on those users<br>
```
#Determine how many unique users does the dataset have. 
#So we concentrate on users that consitute to a large percentage of play counts
#Due to the large size of the file, we will read it line by line
#We will then extract play count information on a user(or song)
output_dict = {}

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
    user_count_df = pd.DataFrame(output_list)
    user_count_df = user_count_df.sort_values(by='play_count',ascending = False)
    user_count_df = user_count_df.reset_index()
    user_count_df.drop(columns = 'index',inplace = True)
    user_count_df.to_csv(path_or_buf = '../data/user_playcount_df.csv',index = False)
```
#### Play counts of some users:<br>
![image](https://user-images.githubusercontent.com/105756607/202028707-97c5b00f-8eaa-4a2d-92fa-0d1c27f6cfca.png)

We will do a similar processing to get play count per song <br>

user_count_df has 1019318 unique users<br>

song_count_df has 384546 unique songs<br>

Finding the number of users that account to 40% of the play counts & Finding the number of songs that account to 80% of the play count
```
#Determining Number of users (n) accounting to 40% of play counts
#Keep changing n until you see 40%
total_play_count = sum(play_count_df.play_count)
(float(play_count_df.head(n=100000).play_count.sum())/total_play_count)*100
```

```
#Determining Number of songs (n) accounting to 80% of play counts
#Keep changing n until you see 80%
total_play_count = sum(song_count_df.play_count)
(float(song_count_df.head(n=30000).play_count.sum())/total_play_count)*100
```

Now that we have discovered that 100000 users account to 40% of the play counts and 30000 songs account to 80% of the play counts, let's create subsets of this data
```
#Subsets of users
user_count_subset = user_count_df.head(n=100000)
user_subset = user_count_subset.user

#Subsets of songs
song_count_subset = song_count_df.head(n=30000)
song_subset = song_count_subset.song
```

We can now subset the original dataset to contain only filtered users and songs.
```
#Code to form subsets of with maximum play counts per song and user
triplet_dataset_sub = triplet_dataset[triplet_dataset.user.isin(user_subset)]
del(triplet_dataset)
triplet_dataset_sub_song = triplet_dataset_sub[triplet_dataset_sub.song.isin(song_subset)]
triplet_dataset_sub_song = triplet_dataset_sub_song.reset_index()
triplet_dataset_sub_song.drop(columns = 'index',inplace = True)
del(triplet_dataset_sub)
```
#### The resultant data subset containing 10,774,558 rows looks like below:
![image](https://user-images.githubusercontent.com/105756607/202034568-a250bede-70d7-450b-abde-0a953ef4a2a4.png)

