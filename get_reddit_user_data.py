import pandas as pd
import numpy as np
import praw
import json
import pymongo


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
    #start api connection and db connection for this process
    reddit = praw.Reddit(client_secret = keys[1],
                     client_id = keys[0] ,
                     user_agent = 'datagathering by /u/GougeC')
    client = pymongo.MongoClient('mongodb://ec2-54-214-228-72.us-west-2.compute.amazonaws.com:27017/')

    users = []
    count = 0

    #loop through users assigned to this function
    for user in user_list:
        count+=1
        #try to get user (make sure it hasn't been deleted)
        try:
            user_object = reddit.redditor(user)
        except:
            print('trouble getting user: ', user)
            continue
        #get user information
        i = 0
        user_dict = {}
        user_dict['username'] = user
        user_dict['subreddits'] = []
        user_dict['comments'] = []
        #get text and subreddit for each of their top twenty comments
        for comment in user_object.comments.hot():
            i+=1
            if i==20: break
            user_dict['subreddits'].append(comment.subreddit)
            user_dict['comments'].append(comment.body)
        users.append(user_dict)

    db = client.capstone_db
    usertable = db.users
    #add each user to the database
    for user in users:
        usertable.insert_one(user)
