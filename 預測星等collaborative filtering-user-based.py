import pandas as pd
review_df=pd.read_json('/Users/wudailing/Desktop/期末報告/yelp_dataset 拷貝/yelp_recruiting/yelp_training_set 2/yelp_training_set_review.json',lines=True)
review_df=review_df[['business_id','user_id','stars']]
review_df=review_df.set_index('user_id',drop=True)
review_df=review_df.pivot_table(index='user_id',columns='business_id',values='stars')
user_df=pd.read_json('/Users/wudailing/Desktop/期末報告/yelp_dataset 拷貝/yelp_recruiting/yelp_training_set 2/yelp_training_set_user.json',lines=True)
user_df=user_df.set_index('user_id',drop=True)['average_stars']
score_df=pd.concat([review_df,user_df],axis=1,sort=False)

FD=pd.read_csv('/Users/wudailing/Desktop/FD_5m.csv')
busi_list=list(FD['business_id'])
user_list=list(FD['user_id'])
#busi_user_dict={'ZRJwVLyzEJq1VAihDhYiow':'0a2KyEL0d3Yb1V6aivbIuQ','6oRAC4uyJCsJl1X0WZpVSA':'0hT2KtfLiobPvh6cDC8JQg','zp713qNhx8d9KCJJnrw1xA':'wFweIWhv2fREZV_dYkz_1g'}

def predfun(busi_id):
    pred_dict = {}
    child2 = 0
    mother2 = 0
    similarity_dict = {}
    for i in range(len(busi_list)):
        userid2=score_df1.index[i]
        user2=score_df1.iloc[i,:]
        user2avg=user2['average_stars']
        dist2=user2[:-1]-user2['average_stars']
        square2=(sum(dist2**2))**(1/2)

        child = sum(dist1 * dist2)
        mother = square1 * square2
        similarity_dict[userid2]=child/mother


        star2=score_df1[busi_id][i]
        child2+=+similarity_dict[score_df1.index[i]]*(star2-user2avg)
        mother2+=similarity_dict[userid2]
    pred_star = targetuser_avg + (child2 / mother2)
    print(pred_star)




for i in range(len(busi_list)):
    busiid=busi_list[i]
    userid=user_list[i]
    score_df1=score_df[score_df[busiid].notna()]
    score_df1=score_df1.fillna(0)
    targetuser=score_df1.loc[userid]
    targetuser_avg=targetuser['average_stars']
    dist1=targetuser[:-1]-targetuser_avg
    square1=(sum(dist1**2))**(1/2)
    score_dict1=dict(score_df1)
    predfun(busiid)










