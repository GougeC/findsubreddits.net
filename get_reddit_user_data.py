import pandas as pd
import numpy as np
import praw
import json

def get_data_for_userlist(user_list,keys,filename):
    '''
    Takes in a list of unique reddit users, and reddit api keys then
    gets a list of their last 20 comments and the subreddits they commented
    in
    (sorted by "hot") if they exist.
    This then writes this info to a json at the filename specified in the
    argument
    this is intended to be ran with multiple processes and api keys
    '''
    reddit = praw.Reddit(client_secret = keys[1],
                     client_id = keys[0] ,
                     user_agent = 'datagathering by /u/GougeC')

    users = {}
    count = 0
    for user in user_list:
        count+=1
        try:
            user_object = reddit.redditor(user)
        except:
            print('trouble getting user: ', user)
            continue
        i = 0
        user_dict = {}
        user_dict['username'] = user
        user_dict['subreddits'] = []
        user_dict['comments'] = []
        for comment in user_object.comments.hot():
            i+=1
            user_dict['subreddits'].append(comment.subreddit)
            user_dict['comments'].append(comment.body)
        users[user] = user_dict
    with open(filename,'w') as f:
        json.dump(users,f)
