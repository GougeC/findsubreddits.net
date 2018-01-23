import pandas as pd
import praw
import json
import time
import numpy as np
from multiprocessing import Process
import sys
import os
import datetime
import requests
from bs4 import BeautifulSoup
import get_reddit_user_data as grud
import pymongo



def do_list_of_subs(subreddit_list,keys,date,user_list):
    '''
    This is the task for each process to complete when multiprocessing data scraping
    from reddit api
    '''
    #start reddit instance
    reddit = praw.Reddit(client_secret = keys[1],
                         client_id = keys[0] ,
                         user_agent = 'datagathering by /u/GougeC')
    #start connection to mongodb database
    client = pymongo.MongoClient('mongodb://ec2-54-214-228-72.us-west-2.compute.amazonaws.com:27017/')
    #loop over each sub assigned to this process
    for sub_name in subreddit_list:
        get_write_sub_data(sub_name,date,reddit,user_list,client)

def get_subreddits():
    '''
    Scrapes redditlist.com for the top 625 subreddits and then
    drops some of the more offensive/inappropriate subs from the list
    before returning a list of the subs
    '''
    subs = []
    for page in range(1,6):
        req = requests.get('http://redditlist.com/?page={}'.format(page))
        soup = BeautifulSoup(req.text,'lxml')
        top = soup.find_all(class_='span4 listing')[1]
        soup2 = top.find_all(class_='subreddit-url')
        for x in soup2:
            soup3 = x.find_all(class_='sfw')[0].text
            subs.append(soup3)
    #subs that I deemed "undesirable" for my recommender (innapropriate or hostile communities)
    drop_list = ['LadyBoners','Celebs', 'pussypassdenied','MensRights','jesuschristreddit','TheRedPill','NoFap']

    for x in subs:
        if x in drop_list:
            subs.remove(x)
    return subs

def get_post_info(post,user_list,subreddit):
    '''
    Input: a PRAW post object
    Output: a post dictionary with data about the post imputed
    including all the top comments and children (up to 1000 comments)
    '''
    post_dict = {}
    post_dict['title'] = post.title
    post_dict['id'] = post.id
    post_dict['permalink'] = post.permalink
    post_dict['subreddit'] = subreddit
    if post.author.name:
        post_dict['author'] = post.author.name
        with open(user_list,'a') as f:
            f.write(post.author.name)
            f.write(',\n')
            f.flush()

    post_dict['selftext'] = post.selftext
    post_dict['domain'] = post.domain
    post_dict['link_url'] = post.url
    comment_list = []
    while len(comment_list) < 1000:
        if not post.comments:
            break
        try:
            post.comments.replace_more()
        except:
            print('failed comments replace_more')
            continue
        try:
            for comment in post.comments:
                if comment.body:
                    comment_list.append(comment.body)
                if comment.author:
                    with open(user_list,'a+') as g:
                        g.write(comment.author.name+',\n')
                        g.flush()
                try:
                    if comment.replies:
                        reps = get_10_children(comment,user_list)
                        comment_list+=reps
                except Exception as e:
                    print('trying to get children broke',str(e))

                if len(comment_list) >= 1000: break
        except:
            print('trying to get comments broke')
            break
    post_dict['comments'] = comment_list
    return post_dict

def get_10_children(comment,user_list):
    '''
    given a reddit comment object in PRAW this returns the text and users
    from 10 children comments
    '''
    comments = []
    i = 0
    #check if the replies exist
    if comment.replies:
        try:
            comment.replies.replace_more()
        except:
            print('comment replace more failed')
        i = 0
        #get data from up to ten replies
        for reply in comment.replies:
            i+=1
            if i==10: break
            if reply.author:
                with open(user_list,'a+') as h:
                    h.write(reply.author.name)
                    h.write(',\n')
                    h.flush()
            if reply.body:
                comments.append(reply.body)
    return comments


def get_write_sub_data(sub_name,date,reddit,user_list,client):
    '''
    Retrives the data from the sub named in the sub_name parameter
    and writes the data as a json with the date included in the filename
    '''
    t1 = time.time()
    print("trying to get ",sub_name)
    #call reddit api to get the sub data
    try:
        sub = reddit.subreddit(sub_name)
    except:
        print('trying to get subbreddit broke',sub_name)
        with open('../data'+date+'/'+'failed_subs'+date+'.txt','a') as f:
            f.write(sub_name+', ')
        return None

    #loop over all the posts in the subreddit object up to 100 posts
    posts = {}
    top = sub.top(time_filter = 'month')
    i = 0
    try:

        for post in top:
            post_dic = {}
            i+=1
            #get metadata from post plus post data from deeper function
            try:
                post_dic['subreddit'] = sub_name
                post_dic['permalink'] = post.permalink
                post_dic['data'] = get_post_info(post,user_list,sub_name)
                #write post data to mongodb
                try:
                    db = client.capstone_db
                    posts = db.posts
                    posts.insert_one(post_dic)
                except Exception as e:
                    print("failed to add post to db ",sub_name, e)
            except Exception as e:
                print('trying get_post_info broke',sub_name, e)
                continue

    except:
        print('trying to loop through posts broke',sub_name)
        return None
            #vestigial code for now
            ##filename = '../data'+date+'/'+sub_name+date+'.json'
            ##print('writing ',sub_name," as ", filename)
            ##subreddit_data = {'subreddit':sub_name,'posts':posts}
            ##db = client.capstone_db
            ##subs = db.subreddits
            ##subs.insert(subreddit_data)
    t2 = time.time()
    elapsed = t2 - t1
    print("finished sub: {}. It took {} seconds.".format(sub_name,elapsed))
if __name__ == "__main__":

    # get subreddits from scraping function
    sublist = get_subreddits()

    #divide list of subs into 4 parts
    n = len(sublist)//4
    print('attempting to get',len(sublist), 'subreddits')
    lists = [sublist[:n],sublist[n:2*n],sublist[2*n:3*n],sublist[3*n:]]
    processes = []
    n = datetime.datetime.now()
    date = "_"+str(n.month)+"_"+str(n.day)
    directory = '../data'+date+'/'

    #make directories
    if not os.path.exists(directory):
        os.makedirs(directory)
    open(directory+'failed_subs'+date+'.txt','w+').close()

    #create each process and assign it its work
    for i in range(1,5):
        keys = np.loadtxt('keys/reddit{}.txt'.format(i),dtype=str,delimiter=',')
        user_list = directory+'users_list'+date+str(i)+'.txt'
        open(user_list,'w+').close()
        p = Process(target=do_list_of_subs, args = (lists[i-1],keys,date,user_list))
        processes.append(p)

    #start and close each process
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print('Processes Finished for subreddit data')

    #read user data that was deposited earlier by each process
    users1 = pd.read_csv(directory+'users_list'+date+str(1)+'.txt',header=None)[0]
    users2 = pd.read_csv(directory+'users_list'+date+str(2)+'.txt',header=None)[0]
    users3 = pd.read_csv(directory+'users_list'+date+str(3)+'.txt',header=None)[0]
    users4 = pd.read_csv(directory+'users_list'+date+str(4)+'.txt',header=None)[0]
    bagousers = set()

    #get a set of the unique users
    for lst in [users1,users2,users3,users4]:
        uniq = lst.unique()
        for user in uniq:
            bagousers.add(user)
    users_unique = list(bagousers)

    #split the user set into 4, for each process
    k = len(users_unique_list)//4
    lists = [users_unique[:k],users_unique[k:2*k],users_unique[2*k,3*k],users_unique[3*k:]]

    #assign each process its work and its api keys
    for i in range(1,5):
        keys = np.loadtxt('keys/reddit{}.txt'.format(i),dtype=str,delimiter=',')
        filename = '../data'+date+'/'+'USER_DATA_'+str(i)+'.json'
        p = Process(target=grud.get_data_for_userlist, args = (lists[i-1],keys,filename,client))
        processes.append(p)

    #start and finish each process
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print('Processes Finished for User Data')
