#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Import all the requisite libraries for extraction
import os
import re 
import wget 
import glob
import time
import selenium
import requests
import numpy as np
import pandas as pd
import zipfile as z
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
import urllib.request as urllib2


# In[12]:


# Get the driver from the location where it has been downloaded 
driver = webdriver.Chrome('#Your path to chromdriver sits here')


# In[13]:


#Specify the URL to be opened for logging into the website
base_url = 'https://www.apmhealtheurope.com/login.php'


# In[14]:


# Open the website using Selenium webdriver
driver.get(base_url)


# In[15]:


# Log-in into the website using the specified credentials 
driver.find_element_by_id('I_login_largeLogin_login_largelogin').send_keys("username")#Type your username here 
driver.find_element_by_id('I_login_largeLogin_login_largepassword').send_keys("password")#Type your password here
driver.find_element_by_id('I_login_largeLogin_login_largeButton_DoLogin_login').click()


# In[16]:


# Select the searchbar and feed in the search keywords
search_sub = driver.find_element_by_id('desktop-search')
search_sub.send_keys('nsclc')
search_sub.submit()


# In[17]:


# Initiate blank lists to store all the information
bool_loop       = True
lst_art_ttl     = []
lst_nkeywords   = []
lst_art_cat     = []
lst_auth        = []
lst_pub_date    = []
lst_art_full    = []
lst_country     = []
lst_art_link    = []


# In[18]:


page = 1


# In[19]:


base_search = 'https://www.apmhealtheurope.com/search.php?mots='
filter_term = 'nsclc'
user_id     = '&uid=#####&page=' #uid = "Type the alotted userid number"
page_no     =  1
search_url_base  = base_search + filter_term + user_id 


# In[20]:


# Exit the loop when condition for boolean gets falsified
while(page_no):
           
    """
    Operation to be done pagewise, first collect all possible links for the given page, filter these links for the title
    of different reports and then loop through these links to extract information and finally move to the next page
    """
    try:
        #Once the breaking condition is achieved, exit the loop
        driver.find_element_by_tag_name('h2')
        print("Search Complete")
        break
    except:
        # Extract all the possible link and get the link corresponding to the stories
        elems = driver.find_elements_by_xpath("//a[@href]")
        link_list = []
        for elem in elems:
            link_list.append(elem.get_attribute("href"))

        # From the extracted possible links, filter out the links based on stories
        blink = []
        for linkl in link_list:
            if 'story' in linkl:
                blink.append(linkl)

        # Loop through all the pages to extract information associated 
        for i in range(len(blink)):
            time.sleep(2)
            # Open the ith link of the website 
            driver.get(blink[i])
            time.sleep(2)
            # Store this link of the story in a list 
            lst_art_link.append(blink[i])
            # Operations for extraction of information from links below
            lst_pub_date.append(driver.find_element_by_class_name("datetime").text) # Date Extraction 
            lst_art_ttl.append(driver.find_element_by_tag_name("h1").text) #Title Extraction
            # Main Text Extraction
            str_art = ''
            for j in driver.find_elements_by_class_name(("paragraphe")):
                str_art += j.text +" " +"<EOP>"+" "
            lst_art_full.append(str_art)
            # Keyword/Tag Extraction
            tag_list = []
            for j in driver.find_elements_by_css_selector('li.tag'):
                if j!= '':
                    tag_list.append(j.text)
            lst_nkeywords.append(tag_list)
            time.sleep(1)
            # Country Exraction: Note this can be done only using XPath and is highly sensitive to changes in the website
            try:
                lst_country.append(driver.find_elements_by_xpath("/html/body/div[5]/div/div/div/div[2]/div[1]/div/div[1]/p/b")[0].text)
            except:
                lst_country.append("NA")
            # Associated category
            cat_list = []
            for j in driver.find_elements_by_css_selector('ul.rubrics'):
                cat_list.append(j.text)
            lst_art_cat.append(cat_list)
            driver.back()
            # Go back to the page containing all the stories
        print("Extraction in progress...")
        page_no = page_no + 1
        new_search_url = base_search + filter_term + user_id + str(page_no)
        time.sleep(5)
        driver.get(new_search_url)


# In[22]:


(len(lst_art_ttl)),(len(lst_nkeywords)),(len(lst_art_cat)),(len(lst_pub_date)),(len(lst_art_full)),(len(lst_country)),(len(lst_art_link))


# In[23]:


apm_europe = pd.DataFrame()
apm_europe['Art_Title']    = lst_art_ttl
apm_europe['Art_Cat']      = lst_art_cat
apm_europe['Art_Date']     = lst_pub_date
apm_europe['Art_Country']  = lst_country
apm_europe['Art_Link']     = lst_art_link
apm_europe['Art_Full']     = lst_art_full
apm_europe['Art_Keywords'] = lst_nkeywords


# In[24]:


apm_europe.to_csv("apm_europe_scraped_nsclc_18032019.csv",index=False)


# In[ ]:




