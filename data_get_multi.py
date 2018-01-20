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



def do_list_of_subs(subreddit_list,keys,date,user_list):
    '''
    This is the task for each process to complete when multiprocessing data scraping
    from reddit api
    '''
    reddit = praw.Reddit(client_secret = keys[1],
                         client_id = keys[0] ,
                         user_agent = 'datagathering by /u/GougeC')

    for sub_name in subreddit_list:
        get_write_sub_data(sub_name,date,reddit,user_list)

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

    drop_list = ['LadyBoners','Celebs', 'pussypassdenied','MensRights','jesuschristreddit','TheRedPill','NoFap']

    for x in subs:
        if x in drop_list:
            subs.remove(x)
    return subs

def get_post_info(post,user_list):
    '''
    Input: a PRAW post object
    Output: a post dictionary with data about the post imputed
    including all the top comments and children (up to 1000 comments)
    '''
    post_dict = {}
    post_dict['title'] = post.title
    post_dict['id'] = post.id
    post_dict['permalink'] = post.permalink
    if post.author:
        post_dict['author'] = post.author.name
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
            print('failed comments.replace_more')

        for comment in post.comments:
            comment_list.append(comment.body)
            if comment.author:
                with open(user_list,'a') as f:
                    f.write(comment.author+', ')
            if comment.replies:
                users,replies = get_10_children(comment)
                comment_list+=replies
                if users:
                    with open(user_list,'a') as f:
                        for user in users:
                            f.write(user+', ')
            if len(comment_list) >= 1000: break
    post_dict['comments'] = comment_list
    return post_dict

def get_10_children(comment,user_list):
    '''
    given a reddit comment object in PRAW this returns the text and users
    from 10 children comments
    '''
    users = []
    comments = []
    i = 0
    if comment.replies:
        comment.replies.replace_more()
        for reply in comment.replies:
            i+=1
            if i==10: break
            if reply.author:
                users.append(reply.author)
            comments.append(reply.body)
    return (users,comments)

def get_write_sub_data(sub_name,date,reddit):
    print(sub_name)
    try:
        sub = reddit.subreddit(sub_name)
    except:
        print('trying to get subbreddit broke',sub_name)
        with open('~/data'+date+'/'+'failed_subs'+date+'.txt','a') as f:
            f.write(sub_name+', ')
        return None


    posts = {}
    top = sub.top(time_filter = 'week')
    i = 0
    try:
        for post in top:
            i+=1
            #totalposts+=1
            #if i%10 == 0:
                #t2 = time.time()
                #elapsed = t2 - t1
                #totaltime+= elapsed
                #print(elapsed, 'seconds taken for last 10')
                #print(totaltime/totalposts,' seconds per post on average')
                #print('getting post number ', i," from r/",sub_name)
                #t1 = time.time()
            try:
                posts[post.id] = get_post_info(post)
            except:
                print('trying get_post_info broke',sub_name,i)
                continue

    except:
        print('trying to loop through posts broke',sub_name)
        return None

    filename = '~/data'+date+'/'+sub_name+date+'.json'
    print('writing ',sub_name," as ", filename)

    with open(filename,'w') as f:
        json.dump(posts,f)


sublist = get_subreddits()
n = len(sublist)//4
lists = [sublist[:n],sublist[n:2*n],sublist[2*n:3*n],sublist[4*n:]]
processes = []
n = datetime.datetime.now()
date = "_"+str(n.month)+"_"+str(n.day)
directory = '~/data'+date+'/'
if not os.path.exists(directory):
    os.makedirs(directory)
user_list = directory+'users_list'+date+'.csv'
open(user_list,'w').close()
open(directory/'failed_subs'+date+'.txt','w').close()
for i in range(1,5):
    keys = np.loadtxt('keys/reddit{}.txt'.format(i),dtype=str,delimiter=',')
    p = Process(target=do_list_of_subs, args = (lists[i-1],keys,date,user_list))
    processes.append(p)
for p in processes:
    p.start()
for p in processes:
    p.join()
