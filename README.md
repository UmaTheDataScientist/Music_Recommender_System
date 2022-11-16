# Music_Recommender_System
![image](https://user-images.githubusercontent.com/105756607/202054681-2ee02dc6-43a6-4b6d-80c7-ee2c8102d124.png)


## Dataset:
<ul>
<li>The million song dataset is a very popular dataset and is available at 
    <a href = "http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/train_triplets.txt.zip">Echnonest Taste Profile Subset</a><br>
<li>The dataset was created as a collaborative project between the Echonest and LABRosa. The dataset we have is only a subset of the million songs dataset.<br>
<li>The dataset contains 48 million lines of triplets. Each triplet contains (user id, song id, play counts).
<li>The overall dataset contains around a million unique users and around 384,000 songs from the million song dataset contained in it.
<li>Upon decompression, the txt file takes a size of 2.79 GB<br>
<li> Place your train_triplets.txt file in the data folder to execute this code.<br>
</ul>

## How to execute
<ol>
    <li>Clone the repository</li>
    <li>Download the <a href = "http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/train_triplets.txt.zip">Echnonest Taste Profile Subset</a>
        and <a href = 'http://millionsongdataset.com/sites/default/files/AdditionalFiles/track_metadata.db'>Song Information</a> files. </li>
        <li>Extract and place the .txt and .db files in the data folder.</li>
    <li>Open the file code/Music_Recommender.ipynb in Jupyter Notebook</li>
    <li>Now, Click on Kernel -> Restart and Run All Cells</li>
    </ol>


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

### Enhancing The Data:
<ul>
<li>Song ID does not give much information. Let's add song name, artist information to our dataset<br>
<li>This data is provided as a SQL database file and is part of the million songs database.<br>
<li> Download the data from <a href = 'http://millionsongdataset.com/sites/default/files/AdditionalFiles/track_metadata.db'>here</a>
<li>Place it in the data folder
</ul>

```
#Finding the tables in the track_metadata.db file
conn = sqlite3.connect('../data/track_metadata.db')
cur = conn.cursor()
cur.execute("select name from sqlite_master where type = 'table'")
cur.fetchall()
```
![image](https://user-images.githubusercontent.com/105756607/202040710-f8d62e26-f10d-4769-8ea0-92656352a924.png)

Since songs is the only table in the track_metadata.db, let's add it to a dataframe
```
track_metadata_df = pd.read_sql_query("SELECT * from songs", conn)
```
Now let's merge the triplet_dataset_sub_song and track_metadata_df. We will also remove unnecessary columns and duplicate songs. The resultant dataset looks like this:
![image](https://user-images.githubusercontent.com/105756607/202042238-69770e03-8fbf-4f4a-a9b2-f26aa2c11330.png)

### Visual Analysis:
Before we start developing the recommendation engine, let's do some visual analysis of our dataset.<br>
We will try to see the different trends in songs, albums, releases (That's what she said! 😜)

#### Most popular songs:
![image](https://user-images.githubusercontent.com/105756607/202046194-034bd533-9a1c-442b-83af-9a36825c3292.png)

You're the one is the most popular song

#### Most popular artists:
![image](https://user-images.githubusercontent.com/105756607/202046624-1f8222a5-6621-4327-9c6c-045037388bc5.png)

Cold Play is the most popular artist

Even though Cold Play is the most popular artist, they don't have a candidate in the most popular song list.<br>

### Recommendation Engine:
The basis of a recommendation engine is always the recorded interction between the users and products.<br>
Different ways of recommending new tracks to different users:
<ol>
    <li><b>User-based recommendation engine:</b> Algorithm will look for similarity among users and will come up with recommendation based on the similarity.</li>        
    <li><b>Content-based recommendation engine:</b> Algorithm will look for features about the content and find similar content. These similarities will be used to make recommendations to end user.</li>
    <li><b>Hybrid-recommendation engine:</b> Alogithm will look for both features of users and content to develop recommendations. Also called Collaborative filtering. They are very effective.</li>
</ol>

### Popularity-Based Recommendation Engine: (User-based)
This is the simplest recommendation engine.<br>
Determine which songs in our dataset have the most users listening to them and then that will become our standard recommendation set for each user.<br>

```
def create_popularity_recommendation(train_data,user_id,item_id):
    #Get a count of user_ids for each unique song as recommendation score
    train_data_grouped = train_data.groupby([item_id]).agg({user_id:'count'}).reset_index()
    train_data_grouped.rename(columns = {user_id:'score'},inplace = True)
    
    #Sort the songs based on recommendation score
    train_data_sort = train_data_grouped.sort_values(['score',item_id],ascending = [0,1])
    
    #Generate a recommendation rank based upon score
    train_data_sort['Rank'] = train_data_sort['score'].rank(ascending = 0,method  ='first')
    
    #Get the top 20 recommendations
    popularity_recommendations = train_data_sort.head(20)
    return popularity_recommendations
```
Any user is most likely to see the following songs in their recommendation if Popularity Based Recommendation Engine is used.

![image](https://user-images.githubusercontent.com/105756607/202052924-964f7ec0-273c-471b-a6d6-072bd8cceed2.png)

However, This is not enough right? Let's get this more personalised!

![image](https://github.com/UmaTheDataScientist/Music_Recommender_System/blob/main/images/not-good.gif)

### Item Similarity Based Recommendation Engine: (Content-Based)
This recommendation engine is based on calculating similarities between user's items and the other items in our dataset.
Similarity between two songs is defined as: if 2 songs are being listened to by a large fraction of common users out of the total listeners, the 2 songs are said to be similar.

```
Similarity(i,j) = intersection(user(i),user(j))/union((user(i),user(j))
```

On the basis of this similarity metric, we can recommend a song to a user k with the following steps:
1. Determine the song listened to by the user k
2. Calculate the similarity of each song in the user's list to those in the dataset.
3. Determine the songs that are most similar to the songs already listened to by the user.
4. Select a subset of these songs as recommendation based on the similarity score.

Since Step 2 is computation heavy, we will select most popular 5000 songs to make computation more feasible.

```
#Subsets of songs
song_count_subset = song_count_df.head(n=5000)
song_subset = song_count_subset.song
triple_dataset_merged_subset = triple_dataset_merged[triple_dataset_merged.song.isin(song_subset)]
```

The Item based recommendation system: (Content-based)
```
train_data,test_data = train_test_split(triple_dataset_merged_subset,test_size = 0.3,random_state = 0)
is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data,'user','title')

#Change the user_id in the square brackets to get recommendation for any of the 100000 users in the dataset
user_id = list(train_data.user)[7]
user_items = is_model.get_user_items(user_id)
is_model.recommend(user_id)
```

The recommendation system is more personalised for each user.

Look at the recommendations we recieved for the user id 7

![image](https://user-images.githubusercontent.com/105756607/202070163-3f21ab70-bfcd-451c-8a21-f3749f82f517.png)

Interesting right? Let's check for user 100

![image](https://user-images.githubusercontent.com/105756607/202074211-9e39f565-b6a4-432d-afe4-8bbf1ff42468.png)

Don't worry if your code takes too long to execute. Mine took about 26 minutes
![image](https://media.tenor.com/uUNv_-QQhTIAAAAd/mrbean-bean.gif)

### Matrix Factorization Based Recommendation Engine: (Hybrid/Collaborative Filtering)

<ul>

<li>These are the most used recommendation systems.</li>
<li>Matrix Factorization is identification of 2 matrices from an initial matrix, such that when these matrices are multiplied, we get the original matrix.</li>
<li>Matrix Factorization helps discover latent features between two different kinds of entities</li>
<li>Latent features here can be soulful lyrics, catchy music, etc</li>
<li>The starting point to matrix factorization is the utility matrix.</li><br>

**Utility Matrix:**

![image](https://user-images.githubusercontent.com/105756607/202080252-391cd8a3-9908-48b7-a685-cf4370df3df8.png)

<li>The utility matrix (U) is a matrix of (user X item) dimension in which each row represents a user and each column stands for an item </li>
<li>The utility matrix (U) usually contains ratings given by users to the various movies.</li>
<li> We can say recommend a movie M5 to a user C if (another user B  likes that movie M5 and user B and user C have given the same rating to movie M4) </li>
</ul>

Now that we have understood the Utility Matrix, let's go further:
<ul>
<li>Matrix Factorization is breaking down Utility Matrix (U) into 2 low rank matrices so that we can recreate U by multiplying those 2 matrices.</li>

![image](https://user-images.githubusercontent.com/105756607/202081291-b41e7cd1-6e58-4e5e-ad46-2b09861ffcc2.png)

<li> The 2 matrices are: A matrix with dimensions of num_users*factors and A matrix with dimensions of factors*num_movies</li>
<li>We will use the most simplest algorithm SVD (Singular Value Decomposition) for determining matrix factorization</li>
<li>SVD return 3 outputs: U,S,V but we need only 2</li>
<li>Since we need only 2, We will reduce S matrix to k components</li>
<li>Compute the square root of reduced matrix Sk to obtain matrix Sk^1/2</li>
<li>The two factorized matrices will now be: U*Sk^1/2,sk^1/2*V</li>
<li>We can generate the prediction of user i for product j by taking the dot product of ith row of first matrix with the jth column of the second matrix</li>
</ul>

<br>

![image](https://media.tenor.com/Tex6pJ7riVsAAAAC/sleepy-yawn.gif)

<br>

Enough Chitchat, let's implement the code:

The first thought in your mind should be - But, we don't have **ratings** for our songs. How would we make the Utility Matrix in the first place?

We do however have play_counts for each song. So we will determine if a user likes a song (strenght of likeness) in the range of [0,1] based on play_counts

```
triple_dataset_merged_sum = triple_dataset_merged[['user','listen_count']].groupby('user').sum().reset_index()
triple_dataset_merged_sum.rename(columns = {'listen_count':'total_listen_count'},inplace = True)
triple_dataset_merged = pd.merge(triple_dataset_merged,triple_dataset_merged_sum)
triple_dataset_merged['fractional_play_count'] = triple_dataset_merged['listen_count']/triple_dataset_merged['total_listen_count']
```

Output of the merged dataframe:
![image](https://user-images.githubusercontent.com/105756607/202114509-3f14b027-e26f-48e5-b03e-08947f5525d6.png)

Next, we have to convert this dataframe into a sparse matrix in the format of utility matrix.

```
small_set = triple_dataset_merged
user_codes = small_set.user.drop_duplicates().reset_index()
song_codes = small_set.song.drop_duplicates().reset_index()
user_codes.rename(columns = {'index':'user_index'},inplace = True)
song_codes.rename(columns = {'index':'song_index'},inplace = True)
user_codes['user_index_value'] = list(user_codes.index)
song_codes['song_index_value'] = list(song_codes.index)
small_set = pd.merge(small_set,song_codes,how = 'left')
small_set = pd.merge(small_set,user_codes,how = 'left')
mat_candidate = small_set[['user_index_value','song_index_value','fractional_play_count']]
data_array = mat_candidate.fractional_play_count.values
row_array = mat_candidate.user_index_value.values
col_array = mat_candidate.song_index_value.values
data_sparse = coo_matrix((data_array,(row_array,col_array)),dtype = float)
```

Our resultant data sparse matrix looks like below:

![image](https://user-images.githubusercontent.com/105756607/202289371-919a8df7-565e-4d37-ad10-be160b628c14.png)


Now that we have our utility matrix, we need to break it down using SVD into 3 different matrices.

```
#Compute SVD of the user ratings matrix
def computeSVD(urm, K):
    U, s, Vt = svds(urm,K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])

    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)
    
    return U, S, Vt
```
Now let's use the above method to get our 3 matrices
We also have to specify the number of latent factors. I'm choosing 50.

```
K = 50
urm = data_sparse
MAX_PID = urm.shape[1]
MAX_UID = urm.shape[0]
U, S, Vt = computeSVD(urm, K)

```

Select users you want the recommendations for 

```
uTest = [4]
print("User id for whom recommendations are needed: %d" % uTest[0])

```

Now let's find recommendations for this user

```
#Compute estimated rating for the test user
def computeEstimatedRatings(urm, U, S, Vt, uTest, K, test):
    rightTerm = S*Vt 

    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm
        #we convert the vector to dense format in order to get the indices 
        #of the movies with the best estimated ratings 
        estimatedRatings[userTest, :] = prod.todense()
        recom = (-estimatedRatings[userTest, :]).argsort()[:250]
    return recom
```

Call this method to get all the recommendations:


```
#Get estimated rating for test user
print("Predictied ratings:")
uTest_recommended_items = computeEstimatedRatings(urm, U, S, Vt, uTest, K, True)
for user in uTest:
    print("Recommendation for user with user id {}".format(user))
    rank_value = 1
    for i in uTest_recommended_items[0:10]:
        song_details = small_set[small_set.song_index_value == i].drop_duplicates('song_index_value')[['title','artist_name']]
        print("The number {} recommended song is {} BY {}".format(rank_value,list(song_details['title'])[0],list(song_details['artist_name'])[0]))
        rank_value+=1
```

The following are the recommendations for the Test User:
![image](https://user-images.githubusercontent.com/105756607/202291405-5f7b81b7-a027-4d40-ac6d-93de35bc3414.png)


