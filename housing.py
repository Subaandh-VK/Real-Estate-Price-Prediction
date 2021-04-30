#!/usr/bin/env python
# coding: utf-8

# # House pricing prediction

# # Data Description

# MSSubClass: Identifies the type of dwelling involved in the sale.	
# 
#         20	1-STORY 1946 & NEWER ALL STYLES
#         30	1-STORY 1945 & OLDER
#         40	1-STORY W/FINISHED ATTIC ALL AGES
#         45	1-1/2 STORY - UNFINISHED ALL AGES
#         50	1-1/2 STORY FINISHED ALL AGES
#         60	2-STORY 1946 & NEWER
#         70	2-STORY 1945 & OLDER
#         75	2-1/2 STORY ALL AGES
#         80	SPLIT OR MULTI-LEVEL
#         85	SPLIT FOYER
#         90	DUPLEX - ALL STYLES AND AGES
#        120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
#        150	1-1/2 STORY PUD - ALL AGES
#        160	2-STORY PUD - 1946 & NEWER
#        180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
#        190	2 FAMILY CONVERSION - ALL STYLES AND AGES
# 
# MSZoning: Identifies the general zoning classification of the sale.
# 		
#        A	Agriculture
#        C	Commercial
#        FV	Floating Village Residential
#        I	Industrial
#        RH	Residential High Density
#        RL	Residential Low Density
#        RP	Residential Low Density Park 
#        RM	Residential Medium Density
# 	
# LotFrontage: Linear feet of street connected to property
# 
# LotArea: Lot size in square feet
# 
# Street: Type of road access to property
# 
#        Grvl	Gravel	
#        Pave	Paved
#        	
# Alley: Type of alley access to property
# 
#        Grvl	Gravel
#        Pave	Paved
#        NA 	No alley access
# 		
# LotShape: General shape of property
# 
#        Reg	Regular	
#        IR1	Slightly irregular
#        IR2	Moderately Irregular
#        IR3	Irregular
#        
# LandContour: Flatness of the property
# 
#        Lvl	Near Flat/Level	
#        Bnk	Banked - Quick and significant rise from street grade to building
#        HLS	Hillside - Significant slope from side to side
#        Low	Depression
# 		
# Utilities: Type of utilities available
# 		
#        AllPub	All public Utilities (E,G,W,& S)	
#        NoSewr	Electricity, Gas, and Water (Septic Tank)
#        NoSeWa	Electricity and Gas Only
#        ELO	Electricity only	
# 	
# LotConfig: Lot configuration
# 
#        Inside	Inside lot
#        Corner	Corner lot
#        CulDSac	Cul-de-sac
#        FR2	Frontage on 2 sides of property
#        FR3	Frontage on 3 sides of property
# 	
# LandSlope: Slope of property
# 		
#        Gtl	Gentle slope
#        Mod	Moderate Slope	
#        Sev	Severe Slope
# 	
# Neighborhood: Physical locations within Ames city limits
# 
#        Blmngtn	Bloomington Heights
#        Blueste	Bluestem
#        BrDale	Briardale
#        BrkSide	Brookside
#        ClearCr	Clear Creek
#        CollgCr	College Creek
#        Crawfor	Crawford
#        Edwards	Edwards
#        Gilbert	Gilbert
#        IDOTRR	Iowa DOT and Rail Road
#        MeadowV	Meadow Village
#        Mitchel	Mitchell
#        Names	North Ames
#        NoRidge	Northridge
#        NPkVill	Northpark Villa
#        NridgHt	Northridge Heights
#        NWAmes	Northwest Ames
#        OldTown	Old Town
#        SWISU	South & West of Iowa State University
#        Sawyer	Sawyer
#        SawyerW	Sawyer West
#        Somerst	Somerset
#        StoneBr	Stone Brook
#        Timber	Timberland
#        Veenker	Veenker
# 			
# Condition1: Proximity to various conditions
# 	
#        Artery	Adjacent to arterial street
#        Feedr	Adjacent to feeder street	
#        Norm	Normal	
#        RRNn	Within 200' of North-South Railroad
#        RRAn	Adjacent to North-South Railroad
#        PosN	Near positive off-site feature--park, greenbelt, etc.
#        PosA	Adjacent to postive off-site feature
#        RRNe	Within 200' of East-West Railroad
#        RRAe	Adjacent to East-West Railroad
# 	
# Condition2: Proximity to various conditions (if more than one is present)
# 		
#        Artery	Adjacent to arterial street
#        Feedr	Adjacent to feeder street	
#        Norm	Normal	
#        RRNn	Within 200' of North-South Railroad
#        RRAn	Adjacent to North-South Railroad
#        PosN	Near positive off-site feature--park, greenbelt, etc.
#        PosA	Adjacent to postive off-site feature
#        RRNe	Within 200' of East-West Railroad
#        RRAe	Adjacent to East-West Railroad
# 	
# BldgType: Type of dwelling
# 		
#        1Fam	Single-family Detached	
#        2FmCon	Two-family Conversion; originally built as one-family dwelling
#        Duplx	Duplex
#        TwnhsE	Townhouse End Unit
#        TwnhsI	Townhouse Inside Unit
# 	
# HouseStyle: Style of dwelling
# 	
#        1Story	One story
#        1.5Fin	One and one-half story: 2nd level finished
#        1.5Unf	One and one-half story: 2nd level unfinished
#        2Story	Two story
#        2.5Fin	Two and one-half story: 2nd level finished
#        2.5Unf	Two and one-half story: 2nd level unfinished
#        SFoyer	Split Foyer
#        SLvl	Split Level
# 	
# OverallQual: Rates the overall material and finish of the house
# 
#        10	Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average
#        5	Average
#        4	Below Average
#        3	Fair
#        2	Poor
#        1	Very Poor
# 	
# OverallCond: Rates the overall condition of the house
# 
#        10	Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average	
#        5	Average
#        4	Below Average	
#        3	Fair
#        2	Poor
#        1	Very Poor
# 		
# YearBuilt: Original construction date
# 
# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
# 
# RoofStyle: Type of roof
# 
#        Flat	Flat
#        Gable	Gable
#        Gambrel	Gabrel (Barn)
#        Hip	Hip
#        Mansard	Mansard
#        Shed	Shed
# 		
# RoofMatl: Roof material
# 
#        ClyTile	Clay or Tile
#        CompShg	Standard (Composite) Shingle
#        Membran	Membrane
#        Metal	Metal
#        Roll	Roll
#        Tar&Grv	Gravel & Tar
#        WdShake	Wood Shakes
#        WdShngl	Wood Shingles
# 		
# Exterior1st: Exterior covering on house
# 
#        AsbShng	Asbestos Shingles
#        AsphShn	Asphalt Shingles
#        BrkComm	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        CemntBd	Cement Board
#        HdBoard	Hard Board
#        ImStucc	Imitation Stucco
#        MetalSd	Metal Siding
#        Other	Other
#        Plywood	Plywood
#        PreCast	PreCast	
#        Stone	Stone
#        Stucco	Stucco
#        VinylSd	Vinyl Siding
#        Wd Sdng	Wood Siding
#        WdShing	Wood Shingles
# 	
# Exterior2nd: Exterior covering on house (if more than one material)
# 
#        AsbShng	Asbestos Shingles
#        AsphShn	Asphalt Shingles
#        BrkComm	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        CemntBd	Cement Board
#        HdBoard	Hard Board
#        ImStucc	Imitation Stucco
#        MetalSd	Metal Siding
#        Other	Other
#        Plywood	Plywood
#        PreCast	PreCast
#        Stone	Stone
#        Stucco	Stucco
#        VinylSd	Vinyl Siding
#        Wd Sdng	Wood Siding
#        WdShing	Wood Shingles
# 	
# MasVnrType: Masonry veneer type
# 
#        BrkCmn	Brick Common
#        BrkFace	Brick Face
#        CBlock	Cinder Block
#        None	None
#        Stone	Stone
# 	
# MasVnrArea: Masonry veneer area in square feet
# 
# ExterQual: Evaluates the quality of the material on the exterior 
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 		
# ExterCond: Evaluates the present condition of the material on the exterior
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 		
# Foundation: Type of foundation
# 		
#        BrkTil	Brick & Tile
#        CBlock	Cinder Block
#        PConc	Poured Contrete	
#        Slab	Slab
#        Stone	Stone
#        Wood	Wood
# 		
# BsmtQual: Evaluates the height of the basement
# 
#        Ex	Excellent (100+ inches)	
#        Gd	Good (90-99 inches)
#        TA	Typical (80-89 inches)
#        Fa	Fair (70-79 inches)
#        Po	Poor (<70 inches
#        NA	No Basement
# 		
# BsmtCond: Evaluates the general condition of the basement
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical - slight dampness allowed
#        Fa	Fair - dampness or some cracking or settling
#        Po	Poor - Severe cracking, settling, or wetness
#        NA	No Basement
# 	
# BsmtExposure: Refers to walkout or garden level walls
# 
#        Gd	Good Exposure
#        Av	Average Exposure (split levels or foyers typically score average or above)	
#        Mn	Mimimum Exposure
#        No	No Exposure
#        NA	No Basement
# 	
# BsmtFinType1: Rating of basement finished area
# 
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
# 		
# BsmtFinSF1: Type 1 finished square feet
# 
# BsmtFinType2: Rating of basement finished area (if multiple types)
# 
#        GLQ	Good Living Quarters
#        ALQ	Average Living Quarters
#        BLQ	Below Average Living Quarters	
#        Rec	Average Rec Room
#        LwQ	Low Quality
#        Unf	Unfinshed
#        NA	No Basement
# 
# BsmtFinSF2: Type 2 finished square feet
# 
# BsmtUnfSF: Unfinished square feet of basement area
# 
# TotalBsmtSF: Total square feet of basement area
# 
# Heating: Type of heating
# 		
#        Floor	Floor Furnace
#        GasA	Gas forced warm air furnace
#        GasW	Gas hot water or steam heat
#        Grav	Gravity furnace	
#        OthW	Hot water or steam heat other than gas
#        Wall	Wall furnace
# 		
# HeatingQC: Heating quality and condition
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# 		
# CentralAir: Central air conditioning
# 
#        N	No
#        Y	Yes
# 		
# Electrical: Electrical system
# 
#        SBrkr	Standard Circuit Breakers & Romex
#        FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
#        FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
#        FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
#        Mix	Mixed
# 		
# 1stFlrSF: First Floor square feet
#  
# 2ndFlrSF: Second floor square feet
# 
# LowQualFinSF: Low quality finished square feet (all floors)
# 
# GrLivArea: Above grade (ground) living area square feet
# 
# BsmtFullBath: Basement full bathrooms
# 
# BsmtHalfBath: Basement half bathrooms
# 
# FullBath: Full bathrooms above grade
# 
# HalfBath: Half baths above grade
# 
# Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
# 
# Kitchen: Kitchens above grade
# 
# KitchenQual: Kitchen quality
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        	
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# 
# Functional: Home functionality (Assume typical unless deductions are warranted)
# 
#        Typ	Typical Functionality
#        Min1	Minor Deductions 1
#        Min2	Minor Deductions 2
#        Mod	Moderate Deductions
#        Maj1	Major Deductions 1
#        Maj2	Major Deductions 2
#        Sev	Severely Damaged
#        Sal	Salvage only
# 		
# Fireplaces: Number of fireplaces
# 
# FireplaceQu: Fireplace quality
# 
#        Ex	Excellent - Exceptional Masonry Fireplace
#        Gd	Good - Masonry Fireplace in main level
#        TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#        Fa	Fair - Prefabricated Fireplace in basement
#        Po	Poor - Ben Franklin Stove
#        NA	No Fireplace
# 		
# GarageType: Garage location
# 		
#        2Types	More than one type of garage
#        Attchd	Attached to home
#        Basment	Basement Garage
#        BuiltIn	Built-In (Garage part of house - typically has room above garage)
#        CarPort	Car Port
#        Detchd	Detached from home
#        NA	No Garage
# 		
# GarageYrBlt: Year garage was built
# 		
# GarageFinish: Interior finish of the garage
# 
#        Fin	Finished
#        RFn	Rough Finished	
#        Unf	Unfinished
#        NA	No Garage
# 		
# GarageCars: Size of garage in car capacity
# 
# GarageArea: Size of garage in square feet
# 
# GarageQual: Garage quality
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
# 		
# GarageCond: Garage condition
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Typical/Average
#        Fa	Fair
#        Po	Poor
#        NA	No Garage
# 		
# PavedDrive: Paved driveway
# 
#        Y	Paved 
#        P	Partial Pavement
#        N	Dirt/Gravel
# 		
# WoodDeckSF: Wood deck area in square feet
# 
# OpenPorchSF: Open porch area in square feet
# 
# EnclosedPorch: Enclosed porch area in square feet
# 
# 3SsnPorch: Three season porch area in square feet
# 
# ScreenPorch: Screen porch area in square feet
# 
# PoolArea: Pool area in square feet
# 
# PoolQC: Pool quality
# 		
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        NA	No Pool
# 		
# Fence: Fence quality
# 		
#        GdPrv	Good Privacy
#        MnPrv	Minimum Privacy
#        GdWo	Good Wood
#        MnWw	Minimum Wood/Wire
#        NA	No Fence
# 	
# MiscFeature: Miscellaneous feature not covered in other categories
# 		
#        Elev	Elevator
#        Gar2	2nd Garage (if not described in garage section)
#        Othr	Other
#        Shed	Shed (over 100 SF)
#        TenC	Tennis Court
#        NA	None
# 		
# MiscVal: $Value of miscellaneous feature
# 
# MoSold: Month Sold (MM)
# 
# YrSold: Year Sold (YYYY)
# 
# SaleType: Type of sale
# 		
#        WD 	Warranty Deed - Conventional
#        CWD	Warranty Deed - Cash
#        VWD	Warranty Deed - VA Loan
#        New	Home just constructed and sold
#        COD	Court Officer Deed/Estate
#        Con	Contract 15% Down payment regular terms
#        ConLw	Contract Low Down payment and low interest
#        ConLI	Contract Low Interest
#        ConLD	Contract Low Down
#        Oth	Other
# 		
# SaleCondition: Condition of sale
# 
#        Normal	Normal Sale
#        Abnorml	Abnormal Sale -  trade, foreclosure, short sale
#        AdjLand	Adjoining Land Purchase
#        Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
#        Family	Sale between family members
#        Partial	Home was not completed when last assessed (associated with New Homes)

# In[67]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[90]:


data = pd.read_csv("train.csv")


# In[91]:


data.head()


# In[92]:


data.shape


# In[93]:


data["Alley"].isnull().sum()


# In[94]:


sns.distplot(data['SalePrice']);


# In[95]:


data.columns


# In[102]:


plt.scatter(x=data['GrLivArea'], y=data['SalePrice']);


# In[103]:


plt.scatter(x=data['OverallQual'], y=data['SalePrice']);


# ## First Model V_0

# In[140]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

X_v0 = data.select_dtypes(include=numerics)


# In[141]:


X_v0.head()


# In[142]:


X_v0.set_index(["Id"])


# In[143]:


X_v0 = X_v0.drop("Id",axis=1)


# In[144]:


fig, axs = plt.subplots(1, figsize=(18,9))
sns.heatmap(X_v0.corr(), ax=axs)


# In[145]:



cols = ['OverallQual', 'YearBuilt','GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath']

fig, axs = plt.subplots(6, figsize=(10,16))
idx = 0

for vals in cols:
    sns.scatterplot(X_v0[vals], data['SalePrice'], ax=axs[idx])
    idx += 1


# In[146]:


X_v0.corr()["SalePrice"]


# In[147]:


X_v0.isnull().sum()


# In[148]:


# Fill with 0
X_v0["GarageYrBlt"] = X_v0["GarageYrBlt"].fillna(value = 0)


# In[149]:


# X_v0["LotFrontage"] = X_v0["LotFrontage"].fillna(method = np.mean(X_v0["LotFrontage"]))


# In[150]:


np.mean(X_v0["LotFrontage"])


# In[151]:


X_v0["LotFrontage"] = X_v0["LotFrontage"].fillna(value = 70.04995836802665)


# In[152]:


X_v0["MasVnrArea"] = X_v0["MasVnrArea"].fillna(value = 0)


# In[153]:


X_v0.head()


# In[154]:


y_v0 = X_v0["SalePrice"]


# In[155]:


X_v0 = X_v0.drop("SalePrice",axis=1)


# In[156]:


from sklearn.model_selection import train_test_split


# In[157]:


X_train, X_test, y_train, y_test = train_test_split(X_v0, y_v0, test_size=0.20, random_state=42)


# ## Linear Regression Model

# In[158]:


from sklearn.linear_model import LinearRegression


# In[159]:


reg = LinearRegression()


# In[160]:


reg.fit(X_train,y_train)


# In[161]:


y_hat = reg.predict(X_test)


# In[162]:


list(zip(y_hat,y_test))


# In[163]:


reg.score(X_test,y_test)


# # RANDOM FOREST

# In[164]:


from sklearn.ensemble import RandomForestRegressor


# In[165]:


regr = RandomForestRegressor(max_depth=2,n_estimators=100)


# In[166]:


regr.fit(X_train,y_train)


# In[167]:


regr.predict(X_test)


# In[168]:


regr.score(X_test,y_test)


# # Grid Search Model

# In[169]:


from sklearn.model_selection import GridSearchCV


# In[170]:


parameters = {'n_estimators':[100,200,150] , 'max_depth':[1, 2,3]}


# In[171]:


regr = RandomForestRegressor()
clf = GridSearchCV(regr, parameters)
clf.fit(X_train, y_train)


# In[172]:


clf.best_estimator_


# In[173]:


regr = RandomForestRegressor(max_depth=3,n_estimators=200)


# In[174]:


regr.fit(X_train,y_train)


# In[175]:


regr.score(X_test,y_test)


# In[176]:


data.head()


# In[177]:


data["Street"].value_counts()


# In[178]:


pd.get_dummies(data["Street"])


# In[179]:


pd.get_dummies(data)


# ### Visualization

# In[180]:


data["Street"].value_counts().plot.pie()


# In[181]:


data["OverallCond"].value_counts().plot.pie()


# In[182]:


data["OverallQual"].value_counts().plot.pie()


# In[183]:


from sklearn.model_selection import GridSearchCV


# In[184]:


l_regr = LinearRegression()
params = {"fit_intercept": [True, False], "normalize": [True, False]}
clf = GridSearchCV(l_regr, params)
clf.fit(X_train, y_train)


# In[185]:


clf.best_estimator_


# In[186]:


lin_reg = LinearRegression(fit_intercept=False, normalize=True)
lin_reg.fit(X_train, y_train)


# In[187]:


lin_reg.predict(X_test)


# In[188]:


lin_reg.score(X_test, y_test)


# ##### Linear regression with grid search CV has a similar result like previous which is 82

# ### Random forest regresson on principal data

# In[189]:


y_p = data['SalePrice']
X_p = data.drop('SalePrice', axis=1)


# In[190]:


X_p.isnull().sum()


# In[191]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

X_p = data.select_dtypes(include=numerics)


X_p.head()


# In[192]:


X_p['LotFrontage'] = X_p["LotFrontage"].fillna(value = 70.04995836802665)
X_p['MasVnrArea'] = X_p["MasVnrArea"].fillna(value = 0)
X_p['GarageYrBlt'] = X_p['GarageYrBlt'].fillna(value=0) 


# In[193]:


y_p.head()


# In[194]:


X_train, X_test, y_train, y_test = train_test_split(X_p, y_p, test_size=0.20, random_state=42)


# In[195]:


principal_parameters = {'n_estimators':[50, 100,150, 200] ,'max_depth':[1, 2,3], 'min_samples_leaf': [1,2]}
regr = RandomForestRegressor()
clf = GridSearchCV(regr, parameters)
clf.fit(X_train, y_train)


# In[196]:


clf.best_estimator_


# In[197]:


rand_regr = RandomForestRegressor(max_depth=3,n_estimators=100, min_samples_leaf=1)


# In[198]:


rand_regr.fit(X_train, y_train)


# In[199]:


rand_regr.score(X_test, y_test)


# #### The score for this model is better 96 compared to the previous score of 79. So there is an improvement of 17.
