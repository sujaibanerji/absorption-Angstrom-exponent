import pandas as pd
from datetime import timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

pm_1_ebc_df = pd.read_csv(r'C:\LocalData\sujaiban\sujai.banerji\Aerosol Optical Properties\beijing_hyytiala_winter_school\2018_2021\hyytiala\ae33_abs_bc_pm1_SMEARii_2018_2021_2.txt', sep = '\s+', header = None)
pm_1_ebc_cols = ['year', 'month', 'date', 'hour', 'minute', 'second', 'pm_1_abs_370', 'pm_1_abs_470', 'pm_1_abs_520', 'pm_1_abs_590', 'pm_1_abs_660', 'pm_1_abs_880', 'pm_1_abs_950', 'pm_1_eBC_370', 'pm_1_eBC_470', 'pm_1_eBC_520', 'pm_1_eBC_590', 'pm_1_eBC_660', 'pm_1_eBC_880', 'pm_1_eBC_950']
pm_1_ebc_np = pm_1_ebc_df.to_numpy()
pm_1_ebc_df = pd.DataFrame(pm_1_ebc_np)
pm_1_ebc_df.columns = pm_1_ebc_cols
pm_1_ebc_datetime = pm_1_ebc_df['year'].astype(int).astype(str) + '-' + pm_1_ebc_df['month'].astype(int).astype(str) + '-' + pm_1_ebc_df['date'].astype(int).astype(str) + ' ' + pm_1_ebc_df['hour'].astype(int).astype(str) + ':' + pm_1_ebc_df['minute'].astype(int).astype(str) + ':' + pm_1_ebc_df['second'].astype(int).astype(str)
pm_1_ebc_datetime = pm_1_ebc_datetime.to_frame()
pm_1_ebc_datetime.columns = ['date_and_time']
pm_1_ebc_df = pm_1_ebc_df.drop(columns = ['year', 'month', 'date', 'hour', 'minute', 'second'])
pm_1_ebc_frames = [pm_1_ebc_datetime, pm_1_ebc_df]
pm_1_ebc_df = pd.concat(pm_1_ebc_frames, axis = 1)
pm_1_ebc_df['date_and_time'] = pd.to_datetime(pm_1_ebc_df['date_and_time'])
pm_1_ebc_df.iloc[:, 1:] = pm_1_ebc_df.iloc[:, 1:].astype(float)

pm_ebc_df_2 = pd.read_csv(r'C:\LocalData\sujaiban\sujai.banerji\Aerosol Optical Properties\beijing_hyytiala_winter_school\2018_2021\hyytiala\ae33_abs_bc_pm1_SMEARii_2022.txt', sep = '\s+', header = None)
ebc_2_datetime_0 = pm_ebc_df_2.iloc [:, 0]
ebc_2_datetime_1 = pm_ebc_df_2.iloc[:, 1]
ebc_2_list = []
ebc_2_list.append(ebc_2_datetime_0)
ebc_2_list.append(ebc_2_datetime_1)
ebc_2_datetime = pd.concat(ebc_2_list, axis = 1)
pm_ebc_list = []
pm_ebc_list.append(ebc_2_datetime_0)
pm_ebc_list.append(ebc_2_datetime_1)
pm_2_ebc_df = pd.concat(pm_ebc_list, axis = 1)
pm_2_ebc_df.columns = ['date', 'time']
pm_2_ebc_df = pd.to_datetime(pm_2_ebc_df['date'] + ' ' + pm_2_ebc_df['time'])
pm_2_ebc_df = pm_2_ebc_df.to_frame()
pm_2_ebc_df.columns = ['date_and_time']
pm_2_ebc_df['date_and_time'] = pd.to_datetime(pm_2_ebc_df['date_and_time'])

ebc_2_df_value = pm_ebc_df_2.iloc[:, 2:]
ebc_2_df_value.columns = ['pm_1_abs_370', 'pm_1_abs_470', 'pm_1_abs_520', 'pm_1_abs_590', 'pm_1_abs_660', 'pm_1_abs_880', 'pm_1_abs_950', 'pm_1_eBC_370', 'pm_1_eBC_470', 'pm_1_eBC_520', 'pm_1_eBC_590', 'pm_1_eBC_660', 'pm_1_eBC_880', 'pm_1_eBC_950']
pm_1_ebc_2 = []
pm_1_ebc_2.append(pm_2_ebc_df)
pm_1_ebc_2.append(ebc_2_df_value)
pm_ebc_2_df = pd.concat(pm_1_ebc_2, axis = 1)

pm_ebc_df_list = []
ebc_1_2_df = pm_1_ebc_df.values
ebc_1_2_df = pd.DataFrame(ebc_1_2_df)
ebc_2_2_df = pm_ebc_2_df.values
ebc_2_2_df = pd.DataFrame(ebc_2_2_df)
pm_ebc_df_list.append(ebc_1_2_df)
pm_ebc_df_list.append(ebc_2_2_df)
pm_1_ebc_df = pd.concat(pm_ebc_df_list)
pm_1_ebc_df.columns = ['date_and_time', 'pm_1_abs_370', 'pm_1_abs_470', 'pm_1_abs_520', 'pm_1_abs_590', 'pm_1_abs_660', 'pm_1_abs_880', 'pm_1_abs_950', 'pm_1_eBC_370', 'pm_1_eBC_470', 'pm_1_eBC_520', 'pm_1_eBC_590', 'pm_1_eBC_660', 'pm_1_eBC_880', 'pm_1_eBC_950']
pm_1_ebc_df['date_and_time'] = pd.to_datetime(pm_1_ebc_df['date_and_time'])
pm_1_ebc_df.iloc[:, 1:] = pm_1_ebc_df.iloc[:, 1:].astype(float)

pm_1_ebc_df = pm_1_ebc_df.resample('h', on = 'date_and_time').mean()
pm_1_ebc_df = pm_1_ebc_df.reset_index()

tol = pd.read_csv(r'C:/LocalData/sujaiban/sujai.banerji/Aerosol Optical Properties/beijing_hyytiala_winter_school/2018_2021/hyytiala/ToL_hyytiala_2018_2022.csv')
tol_cols = ['date_and_time', 'sector', 'tol']
tol.columns = tol_cols
tol['date_and_time'] = pd.to_datetime(tol['date_and_time'])
tol['date_and_time'] = tol['date_and_time'] + timedelta(hours = 2)
tol = tol.resample('h', on = 'date_and_time').mean()
tol = tol.reset_index()
tol = tol[tol['sector'] == 1]
tol = tol.iloc[:, [0, 2]]

pm_1_tol_df = pd.merge(pm_1_ebc_df, tol, on = 'date_and_time', how = 'outer')
pm_1_tol_df = pm_1_tol_df.resample('h', on = 'date_and_time').mean()
pm_1_tol_df = pm_1_tol_df.reset_index()

start_year_minus_one = pm_1_tol_df.iloc[0, 0].year - 1
end_year_plus_one = pm_1_tol_df.iloc[-1, 0].year + 1
time_step = 1

pm_1_tol_winter = []
pm_1_tol_spring = []
pm_1_tol_summer = []
pm_1_tol_autumn = []

for i in range(start_year_minus_one, end_year_plus_one, time_step):
    pm_1_tol_dec = pm_1_tol_df[(pm_1_tol_df.iloc[:, 0].dt.year == i) & (pm_1_tol_df.iloc[:, 0].dt.month == 12)]
    pm_1_tol_jan = pm_1_tol_df[(pm_1_tol_df.iloc[:, 0].dt.year == i + 1) & (pm_1_tol_df.iloc[:, 0].dt.month == 1)]
    pm_1_tol_feb = pm_1_tol_df[(pm_1_tol_df.iloc[:, 0].dt.year == i + 1) & (pm_1_tol_df.iloc[:, 0].dt.month == 2)]
    pm_1_tol_winter.append(pm_1_tol_dec)
    pm_1_tol_winter.append(pm_1_tol_jan)
    pm_1_tol_winter.append(pm_1_tol_feb)
    pm_1_tol_mar = pm_1_tol_df[(pm_1_tol_df.iloc[:, 0].dt.year == i) & (pm_1_tol_df.iloc[:, 0].dt.month == 3)]
    pm_1_tol_apr = pm_1_tol_df[(pm_1_tol_df.iloc[:, 0].dt.year == i) & (pm_1_tol_df.iloc[:, 0].dt.month == 4)]
    pm_1_tol_may = pm_1_tol_df[(pm_1_tol_df.iloc[:, 0].dt.year == i) & (pm_1_tol_df.iloc[:, 0].dt.month == 5)]
    pm_1_tol_spring.append(pm_1_tol_mar)
    pm_1_tol_spring.append(pm_1_tol_apr)
    pm_1_tol_spring.append(pm_1_tol_may)
    pm_1_tol_jun = pm_1_tol_df[(pm_1_tol_df.iloc[:, 0].dt.year == i) & (pm_1_tol_df.iloc[:, 0].dt.month == 6)]
    pm_1_tol_jul = pm_1_tol_df[(pm_1_tol_df.iloc[:, 0].dt.year == i) & (pm_1_tol_df.iloc[:, 0].dt.month == 7)]
    pm_1_tol_aug = pm_1_tol_df[(pm_1_tol_df.iloc[:, 0].dt.year == i) & (pm_1_tol_df.iloc[:, 0].dt.month == 8)]
    pm_1_tol_summer.append(pm_1_tol_jun)
    pm_1_tol_summer.append(pm_1_tol_jul)
    pm_1_tol_summer.append(pm_1_tol_aug)
    pm_1_tol_sep = pm_1_tol_df[(pm_1_tol_df.iloc[:, 0].dt.year == i) & (pm_1_tol_df.iloc[:, 0].dt.month == 9)]
    pm_1_tol_oct = pm_1_tol_df[(pm_1_tol_df.iloc[:, 0].dt.year == i) & (pm_1_tol_df.iloc[:, 0].dt.month == 10)]
    pm_1_tol_nov = pm_1_tol_df[(pm_1_tol_df.iloc[:, 0].dt.year == i) & (pm_1_tol_df.iloc[:, 0].dt.month == 11)]
    pm_1_tol_autumn.append(pm_1_tol_sep)
    pm_1_tol_autumn.append(pm_1_tol_oct)
    pm_1_tol_autumn.append(pm_1_tol_nov)
    
pm_1_tol_winter = pd.concat(pm_1_tol_winter)
pm_1_tol_spring = pd.concat(pm_1_tol_spring)
pm_1_tol_summer = pd.concat(pm_1_tol_summer)
pm_1_tol_autumn = pd.concat(pm_1_tol_autumn)

pm_1_tol_winter = pm_1_tol_winter.dropna()

wavelengths = [370, 470, 520, 590, 660, 880, 950]
x = pd.DataFrame(np.log(wavelengths))
x = x.dropna()
x = x.replace([np.inf, -np.inf], np.nan).dropna()
x = x.transpose()
y = pd.DataFrame(np.log(pm_1_tol_winter.iloc[:, 1:8]))
y = y.dropna()
y = y.replace([np.inf, -np.inf], np.nan).dropna()
pm_1_tol_winter = pm_1_tol_winter.loc[y.index]
y = y.transpose()
model = LinearRegression()
slope = pd.DataFrame({'slope': [(model.fit(x.values.reshape(-1, 1), y.iloc[:, idx]).coef_[0]) for idx in range(y.shape[1])]})
tol_winter_subset_1 = pm_1_tol_winter.iloc[:, :15]
tol_winter_subset_1 = tol_winter_subset_1.reset_index()
tol_winter_subset_1 = tol_winter_subset_1.iloc[:, 1:]
slope = slope.reset_index()
slope = slope.iloc[:, 1]
slope = - slope
tol_winter_subset_2 = pm_1_tol_winter.iloc[:, 15]
tol_winter_subset_2 = tol_winter_subset_2.reset_index()
tol_winter_subset_2 = tol_winter_subset_2.iloc[:, 1]
tol_winter_list = []
tol_winter_list.append(tol_winter_subset_1)
tol_winter_list.append(slope)
tol_winter_list.append(tol_winter_subset_2)
pm_1_tol_winter = pd.concat(tol_winter_list, axis = 1) 

winter_aae_tol_hour = []
winter_aae_tol_year = []
start_year_minus_one = pm_1_tol_winter.iloc[0, 0].year - 1
end_year_plus_one = pm_1_tol_winter.iloc[-1, 0].year + 1
time_step = 1

diurnal_aae_tol_row = []

for i in range (start_year_minus_one, end_year_plus_one):
    diurnal_aae_tol_row.append(i)
    
diurnal_aae_tol_winter = []

start_year_minus_one = pm_1_tol_winter.iloc[0, 0].year - 1
end_year_plus_one = pm_1_tol_winter.iloc[-1, 0].year + 1
time_step = 1

start_ah = pm_1_tol_winter.iloc[:, -1].min()
end_ah_plus_one = pm_1_tol_winter.iloc[:, -1].max() + 1
time_ah = 1

for i in range(start_year_minus_one, end_year_plus_one, time_step):
    for j in range(int(start_ah), int(end_ah_plus_one), int(time_ah)):
        diurnal_aae_tol_jan = pm_1_tol_winter[(pm_1_tol_winter.iloc[:, 0].dt.year == i) & (pm_1_tol_winter.iloc[:, 0].dt.month == 12) & (pm_1_tol_winter.iloc[:, -1] == j)]
        diurnal_aae_tol_feb = pm_1_tol_winter[(pm_1_tol_winter.iloc[:, 0].dt.year == i + 1) & (pm_1_tol_winter.iloc[:, 0].dt.month == 1) & (pm_1_tol_winter.iloc[:, -1] == j)]
        diurnal_aae_tol_mar = pm_1_tol_winter[(pm_1_tol_winter.iloc[:, 0].dt.year == i + 1) & (pm_1_tol_winter.iloc[:, 0].dt.month == 2) & (pm_1_tol_winter.iloc[:, -1] == j)]
        diurnal_aae_tol_winter.append(diurnal_aae_tol_jan)
        diurnal_aae_tol_winter.append(diurnal_aae_tol_feb)
        diurnal_aae_tol_winter.append(diurnal_aae_tol_mar)
        diurnal_aae_tol_row = pd.concat(diurnal_aae_tol_winter)
        diurnal_aae_tol_winter = []
        winter_aae_tol_value = diurnal_aae_tol_row.iloc[:, -2]
        winter_aae_tol_value = winter_aae_tol_value.to_frame()
        winter_aae_tol_hour.append(winter_aae_tol_value)
    winter_aae_tol_value = pd.concat(winter_aae_tol_hour, axis = 1)
    winter_aae_tol_hour = []
    winter_aae_tol_year.append(winter_aae_tol_value)
    
diurnal_tol_winter_row = pd.concat(winter_aae_tol_year)

diurnal_aae_tol_column = []

for i in range(int(start_ah), int(end_ah_plus_one), int(time_ah)):
    diurnal_aae_tol_column.append(i)
    
diurnal_tol_winter_row.columns = diurnal_aae_tol_column

tol_hour = 90
tol_5_h_group = tol_hour // 5

group_5_h_index = [range(i*5, (i + 1)*5) for i in range(tol_5_h_group)]

diurnal_tol_winter_average = pd.concat([diurnal_tol_winter_row.iloc[:, indices].mean(axis = 1) for indices in group_5_h_index], axis = 1)

old_column_name = diurnal_tol_winter_row.columns
new_column_name = [str(int((int(old_column_name[i]) + int(old_column_name[i + 4])) / 2)) for i in range(0, tol_hour, 5)]
diurnal_tol_winter_average.columns = new_column_name

plt.title('PM$_1$') 
diurnal_tol_winter_average.boxplot(showfliers = False, color = 'tab:blue', boxprops = {'linewidth': 2, 'color': 'tab:blue'}, medianprops = {'linewidth': 2, 'color': 'tab:blue'}, whiskerprops = {'linewidth': 2, 'color': 'tab:blue'}, capprops = {'linewidth': 2, 'color': 'tab:blue'}, whis = [10, 90], rot = 90) 
plt.xlabel('Time over land (hour)')
plt.ylabel('AAE')
plt.legend(['Winter season'], bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()

winter_aae_tol_hour = []
winter_aae_tol_year = []
start_year_minus_one = pm_1_tol_winter.iloc[0, 0].year - 1
end_year_plus_one = pm_1_tol_winter.iloc[-1, 0].year + 1
time_step = 1

diurnal_aae_tol_row = []

for i in range (start_year_minus_one, end_year_plus_one, time_step):
     diurnal_aae_tol_row.append(i)

for i in range(start_year_minus_one, end_year_plus_one, time_step):
    for j in range(0, 24, 1):
        diurnal_aae_tol_winter = pm_1_tol_winter[(pm_1_tol_winter.iloc[:, 0].dt.year == i) & (pm_1_tol_winter.iloc[:, 0].dt.month == 12) & (pm_1_tol_winter.iloc[:, 0].dt.hour == j) | (pm_1_tol_winter.iloc[:, 0].dt.year == i + 1) & (pm_1_tol_winter.iloc[:, 0].dt.month == 1) & (pm_1_tol_winter.iloc[:, 0].dt.hour == j) | (pm_1_tol_winter.iloc[:, 0].dt.year == i + 1) & (pm_1_tol_winter.iloc[:, 0].dt.month == 2) & (pm_1_tol_winter.iloc[:, 0].dt.hour == j)]
        winter_aae_tol_value = diurnal_aae_tol_winter.iloc[:, -2]
        winter_aae_tol_value = winter_aae_tol_value.to_frame()
        winter_aae_tol_value = winter_aae_tol_value.median()
        winter_aae_tol_hour.append(winter_aae_tol_value)
    winter_aae_tol_value = pd.concat(winter_aae_tol_hour, axis = 1)
    winter_aae_tol_hour = []
    winter_aae_tol_year.append(winter_aae_tol_value)

diurnal_aae_tol_winter = pd.concat(winter_aae_tol_year)
diurnal_aae_tol_column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
diurnal_aae_tol_winter.columns = diurnal_aae_tol_column
diurnal_aae_tol_winter = diurnal_aae_tol_winter.reset_index(drop = True)
diurnal_aae_tol_winter.index = diurnal_aae_tol_row
diurnal_aae_tol_winter = diurnal_aae_tol_winter.T
diurnal_aae_tol_winter = diurnal_aae_tol_winter.dropna(axis = 1)
diurnal_aae_tol_row = []
diurnal_aae_tol_row = diurnal_aae_tol_winter.columns

diurnal_aae_tol_winter.plot(marker = 's')
plt.title('PM$_1$')
plt.ylabel('AAE') 
plt.xlabel('Hour')
plt.legend(diurnal_aae_tol_row, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()

plt.title('PM$_1$') 
diurnal_aae_tol_winter.boxplot(showfliers = False, color = 'tab:blue', boxprops = {'linewidth': 2, 'color': 'tab:blue'}, medianprops = {'linewidth': 2, 'color': 'tab:blue'}, whiskerprops = {'linewidth': 2, 'color': 'tab:blue'}, capprops = {'linewidth': 2, 'color': 'tab:blue'}, whis = [10, 90], rot = 90) 
plt.xlabel('Year)')
plt.ylabel('AAE')
plt.show()

pm_1_tol_spring = pm_1_tol_spring.dropna()

wavelengths = [370, 470, 520, 590, 660, 880, 950]
x = pd.DataFrame(np.log(wavelengths))
x = x.dropna()
x = x.replace([np.inf, -np.inf], np.nan).dropna()
x = x.transpose()
y = pd.DataFrame(np.log(pm_1_tol_spring.iloc[:, 1:8]))
y = y.dropna()
y = y.replace([np.inf, -np.inf], np.nan).dropna()
pm_1_tol_spring = pm_1_tol_spring.loc[y.index]
y = y.transpose()
model = LinearRegression()
slope = pd.DataFrame({'slope': [(model.fit(x.values.reshape(-1, 1), y.iloc[:, idx]).coef_[0]) for idx in range(y.shape[1])]})
tol_spring_subset_1 = pm_1_tol_spring.iloc[:, :15]
tol_spring_subset_1 = tol_spring_subset_1.reset_index()
tol_spring_subset_1 = tol_spring_subset_1.iloc[:, 1:]
slope = slope.reset_index()
slope = slope.iloc[:, 1]
slope = - slope
tol_spring_subset_2 = pm_1_tol_spring.iloc[:, 15]
tol_spring_subset_2 = tol_spring_subset_2.reset_index()
tol_spring_subset_2 = tol_spring_subset_2.iloc[:, 1]
tol_spring_list = []
tol_spring_list.append(tol_spring_subset_1)
tol_spring_list.append(slope)
tol_spring_list.append(tol_spring_subset_2)
pm_1_tol_spring = pd.concat(tol_spring_list, axis = 1) 

spring_aae_tol_hour = []
spring_aae_tol_year = []
start_year = pm_1_tol_spring.iloc[0, 0].year
end_year_plus_one = pm_1_tol_spring.iloc[-1, 0].year + 1
time_step = 1

diurnal_aae_tol_row = []

for i in range (start_year, end_year_plus_one):
    diurnal_aae_tol_row.append(i)
    
diurnal_aae_tol_spring = []

start_year = pm_1_tol_spring.iloc[0, 0].year
end_year_plus_one = pm_1_tol_spring.iloc[-1, 0].year + 1
time_step = 1

start_ah = pm_1_tol_spring.iloc[:, -1].min()
end_ah_plus_one = pm_1_tol_spring.iloc[:, -1].max() + 1
time_ah = 1

for i in range(start_year, end_year_plus_one, time_step):
    for j in range(int(start_ah), int(end_ah_plus_one), int(time_ah)):
        diurnal_aae_tol_mar = pm_1_tol_spring[(pm_1_tol_spring.iloc[:, 0].dt.year == i) & (pm_1_tol_spring.iloc[:, 0].dt.month == 3) & (pm_1_tol_spring.iloc[:, -1] == j)]
        diurnal_aae_tol_apr = pm_1_tol_spring[(pm_1_tol_spring.iloc[:, 0].dt.year == i) & (pm_1_tol_spring.iloc[:, 0].dt.month == 4) & (pm_1_tol_spring.iloc[:, -1] == j)]
        diurnal_aae_tol_may = pm_1_tol_spring[(pm_1_tol_spring.iloc[:, 0].dt.year == i) & (pm_1_tol_spring.iloc[:, 0].dt.month == 5) & (pm_1_tol_spring.iloc[:, -1] == j)]
        diurnal_aae_tol_spring.append(diurnal_aae_tol_mar)
        diurnal_aae_tol_spring.append(diurnal_aae_tol_apr)
        diurnal_aae_tol_spring.append(diurnal_aae_tol_may)
        diurnal_aae_tol_row = pd.concat(diurnal_aae_tol_spring)
        diurnal_aae_tol_spring = []
        spring_aae_tol_value = diurnal_aae_tol_row.iloc[:, -2]
        spring_aae_tol_value = spring_aae_tol_value.to_frame()
        spring_aae_tol_hour.append(spring_aae_tol_value)
    spring_aae_tol_value = pd.concat(spring_aae_tol_hour, axis = 1)
    spring_aae_tol_hour = []
    spring_aae_tol_year.append(spring_aae_tol_value)
    
diurnal_tol_spring_row = pd.concat(spring_aae_tol_year)

diurnal_aae_tol_column = []

for i in range(int(start_ah), int(end_ah_plus_one), int(time_ah)):
    diurnal_aae_tol_column.append(i)
    
diurnal_tol_spring_row.columns = diurnal_aae_tol_column

tol_hour = 90
tol_5_h_group = tol_hour // 5

group_5_h_index = [range(i*5, (i + 1)*5) for i in range(tol_5_h_group)]

diurnal_tol_spring_average = pd.concat([diurnal_tol_spring_row.iloc[:, indices].mean(axis = 1) for indices in group_5_h_index], axis = 1)

old_column_name = diurnal_tol_spring_row.columns
new_column_name = [str(int((int(old_column_name[i]) + int(old_column_name[i + 4])) / 2)) for i in range(0, tol_hour, 5)]
diurnal_tol_spring_average.columns = new_column_name

plt.title('PM$_1$') 
diurnal_tol_spring_average.boxplot(showfliers = False, color = 'tab:green', boxprops = {'linewidth': 2, 'color': 'tab:green'}, medianprops = {'linewidth': 2, 'color': 'tab:green'}, whiskerprops = {'linewidth': 2, 'color': 'tab:green'}, capprops = {'linewidth': 2, 'color': 'tab:green'}, whis = [10, 90], rot = 90) 
plt.xlabel('Time over land (hour)')
plt.ylabel('AAE')
plt.legend(['Spring season'], bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()

spring_aae_tol_hour = []
spring_aae_tol_year = []
start_year  = pm_1_tol_spring.iloc[0, 0].year
end_year_plus_one = pm_1_tol_spring.iloc[-1, 0].year + 1
time_step = 1

diurnal_aae_tol_row = []

for i in range (start_year, end_year_plus_one, time_step):
     diurnal_aae_tol_row.append(i)

for i in range(start_year, end_year_plus_one, time_step):
    for j in range(0, 24, 1):
        diurnal_aae_tol_spring = pm_1_tol_spring[(pm_1_tol_spring.iloc[:, 0].dt.year == i) & (pm_1_tol_spring.iloc[:, 0].dt.month == 3) & (pm_1_tol_spring.iloc[:, 0].dt.hour == j) | (pm_1_tol_spring.iloc[:, 0].dt.year == i) & (pm_1_tol_spring.iloc[:, 0].dt.month == 4) & (pm_1_tol_spring.iloc[:, 0].dt.hour == j) | (pm_1_tol_spring.iloc[:, 0].dt.year == i) & (pm_1_tol_spring.iloc[:, 0].dt.month == 5) & (pm_1_tol_spring.iloc[:, 0].dt.hour == j)]
        spring_aae_tol_value = diurnal_aae_tol_spring.iloc[:, -2]
        spring_aae_tol_value = spring_aae_tol_value.to_frame()
        spring_aae_tol_value = spring_aae_tol_value.median()
        spring_aae_tol_hour.append(spring_aae_tol_value)
    spring_aae_tol_value = pd.concat(spring_aae_tol_hour, axis = 1)
    spring_aae_tol_hour = []
    spring_aae_tol_year.append(spring_aae_tol_value)

diurnal_aae_tol_spring = pd.concat(spring_aae_tol_year)
diurnal_aae_tol_column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
diurnal_aae_tol_spring.columns = diurnal_aae_tol_column
diurnal_aae_tol_spring = diurnal_aae_tol_spring.reset_index(drop = True)
diurnal_aae_tol_spring.index = diurnal_aae_tol_row
diurnal_aae_tol_spring = diurnal_aae_tol_spring.T
diurnal_aae_tol_spring = diurnal_aae_tol_spring.dropna(axis = 1)
diurnal_aae_tol_row = []
diurnal_aae_tol_row = diurnal_aae_tol_spring.columns

diurnal_aae_tol_spring.plot(marker = 's')
plt.title('PM$_1$')
plt.ylabel('AAE') 
plt.xlabel('Hour')
plt.legend(diurnal_aae_tol_row, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()

plt.title('PM$_1$') 
diurnal_aae_tol_spring.boxplot(showfliers = False, color = 'tab:green', boxprops = {'linewidth': 2, 'color': 'tab:green'}, medianprops = {'linewidth': 2, 'color': 'tab:green'}, whiskerprops = {'linewidth': 2, 'color': 'tab:green'}, capprops = {'linewidth': 2, 'color': 'tab:green'}, whis = [10, 90], rot = 90) 
plt.xlabel('Year)')
plt.ylabel('AAE')
plt.show()

pm_1_tol_summer = pm_1_tol_summer.dropna()

wavelengths = [370, 470, 520, 590, 660, 880, 950]
x = pd.DataFrame(np.log(wavelengths))
x = x.dropna()
x = x.replace([np.inf, -np.inf], np.nan).dropna()
x = x.transpose()
y = pd.DataFrame(np.log(pm_1_tol_summer.iloc[:, 1:8]))
y = y.dropna()
y = y.replace([np.inf, -np.inf], np.nan).dropna()
pm_1_tol_summer = pm_1_tol_summer.loc[y.index]
y = y.transpose()
model = LinearRegression()
slope = pd.DataFrame({'slope': [(model.fit(x.values.reshape(-1, 1), y.iloc[:, idx]).coef_[0]) for idx in range(y.shape[1])]})
tol_summer_subset_1 = pm_1_tol_summer.iloc[:, :15]
tol_summer_subset_1 = tol_summer_subset_1.reset_index()
tol_summer_subset_1 = tol_summer_subset_1.iloc[:, 1:]
slope = slope.reset_index()
slope = slope.iloc[:, 1]
slope = - slope
tol_summer_subset_2 = pm_1_tol_summer.iloc[:, 15]
tol_summer_subset_2 = tol_summer_subset_2.reset_index()
tol_summer_subset_2 = tol_summer_subset_2.iloc[:, 1]
tol_summer_list = []
tol_summer_list.append(tol_summer_subset_1)
tol_summer_list.append(slope)
tol_summer_list.append(tol_summer_subset_2)
pm_1_tol_summer = pd.concat(tol_summer_list, axis = 1) 

summer_aae_tol_hour = []
summer_aae_tol_year = []
start_year = pm_1_tol_summer.iloc[0, 0].year
end_year_plus_one = pm_1_tol_summer.iloc[-1, 0].year + 1
time_step = 1

diurnal_aae_tol_row = []

for i in range (start_year, end_year_plus_one):
    diurnal_aae_tol_row.append(i)
    
diurnal_aae_tol_summer = []

start_year = pm_1_tol_summer.iloc[0, 0].year
end_year_plus_one = pm_1_tol_summer.iloc[-1, 0].year + 1
time_step = 1

start_ah = pm_1_tol_summer.iloc[:, -1].min()
end_ah_plus_one = pm_1_tol_summer.iloc[:, -1].max() + 1
time_ah = 1

for i in range(start_year, end_year_plus_one, time_step):
    for j in range(int(start_ah), int(end_ah_plus_one), int(time_ah)):
        diurnal_aae_tol_jun = pm_1_tol_summer[(pm_1_tol_summer.iloc[:, 0].dt.year == i) & (pm_1_tol_summer.iloc[:, 0].dt.month == 6) & (pm_1_tol_summer.iloc[:, -1] == j)]
        diurnal_aae_tol_jul = pm_1_tol_summer[(pm_1_tol_summer.iloc[:, 0].dt.year == i) & (pm_1_tol_summer.iloc[:, 0].dt.month == 7) & (pm_1_tol_summer.iloc[:, -1] == j)]
        diurnal_aae_tol_aug = pm_1_tol_summer[(pm_1_tol_summer.iloc[:, 0].dt.year == i) & (pm_1_tol_summer.iloc[:, 0].dt.month == 8) & (pm_1_tol_summer.iloc[:, -1] == j)]
        diurnal_aae_tol_summer.append(diurnal_aae_tol_jun)
        diurnal_aae_tol_summer.append(diurnal_aae_tol_jul)
        diurnal_aae_tol_summer.append(diurnal_aae_tol_aug)
        diurnal_aae_tol_row = pd.concat(diurnal_aae_tol_summer)
        diurnal_aae_tol_summer = []
        summer_aae_tol_value = diurnal_aae_tol_row.iloc[:, -2]
        summer_aae_tol_value = summer_aae_tol_value.to_frame()
        summer_aae_tol_hour.append(summer_aae_tol_value)
    summer_aae_tol_value = pd.concat(summer_aae_tol_hour, axis = 1)
    summer_aae_tol_hour = []
    summer_aae_tol_year.append(summer_aae_tol_value)
    
diurnal_tol_summer_row = pd.concat(summer_aae_tol_year)

diurnal_aae_tol_column = []

for i in range(int(start_ah), int(end_ah_plus_one), int(time_ah)):
    diurnal_aae_tol_column.append(i)

diurnal_tol_summer_row.columns = diurnal_aae_tol_column

num_cols = len(diurnal_tol_summer_row.columns)
tol_5_h_group = num_cols // 5 + 1  # adjust range to ensure last index is within bounds

group_5_h_index = [range(i*5, min((i + 1)*5, num_cols)) for i in range(tol_5_h_group)]

diurnal_tol_summer_average = pd.concat([diurnal_tol_summer_row.iloc[:, indices].mean(axis=1) for indices in group_5_h_index], axis=1)

if len(old_column_name) % 5 != 0:
    old_column_name = old_column_name[:-1]
new_column_name = [str(int((int(old_column_name[i]) + int(old_column_name[i + 4])) / 2)) for i in range(0, len(old_column_name) - 1, 5)]

tol_hour = 90
tol_5_h_group = tol_hour // 5

group_5_h_index = [range(i*5, (i + 1)*5) for i in range(tol_5_h_group)]

diurnal_tol_spring_average = pd.concat([diurnal_tol_spring_row.iloc[:, indices].mean(axis = 1) for indices in group_5_h_index], axis = 1)

old_column_name = diurnal_tol_spring_row.columns
new_column_name = [str(int((int(old_column_name[i]) + int(old_column_name[i + 4])) / 2)) for i in range(0, tol_hour, 5)]

diurnal_tol_summer_average.columns = new_column_name

plt.title('PM$_1$') 
diurnal_tol_summer_average.boxplot(showfliers = False, color = 'tab:red', boxprops = {'linewidth': 2, 'color': 'tab:red'}, medianprops = {'linewidth': 2, 'color': 'tab:red'}, whiskerprops = {'linewidth': 2, 'color': 'tab:red'}, capprops = {'linewidth': 2, 'color': 'tab:red'}, whis = [10, 90], rot = 90) 
plt.xlabel('Time over land (hour)')
plt.ylabel('AAE')
plt.legend(['Summer season'], bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()

summer_aae_tol_hour = []
summer_aae_tol_year = []
start_year = pm_1_tol_summer.iloc[0, 0].year
end_year_plus_one = pm_1_tol_summer.iloc[-1, 0].year + 1
time_step = 1

diurnal_aae_tol_row = []

for i in range (start_year, end_year_plus_one, time_step):
     diurnal_aae_tol_row.append(i)

for i in range(start_year, end_year_plus_one, time_step):
    for j in range(0, 24, 1):
        diurnal_aae_tol_summer = pm_1_tol_summer[(pm_1_tol_summer.iloc[:, 0].dt.year == i) & (pm_1_tol_summer.iloc[:, 0].dt.month == 6) & (pm_1_tol_summer.iloc[:, 0].dt.hour == j) | (pm_1_tol_summer.iloc[:, 0].dt.year == i) & (pm_1_tol_summer.iloc[:, 0].dt.month == 7) & (pm_1_tol_summer.iloc[:, 0].dt.hour == j) | (pm_1_tol_summer.iloc[:, 0].dt.year == i) & (pm_1_tol_summer.iloc[:, 0].dt.month == 8) & (pm_1_tol_summer.iloc[:, 0].dt.hour == j)]
        summer_aae_tol_value = diurnal_aae_tol_summer.iloc[:, -2]
        summer_aae_tol_value = summer_aae_tol_value.to_frame()
        summer_aae_tol_value = summer_aae_tol_value.median()
        summer_aae_tol_hour.append(summer_aae_tol_value)
    summer_aae_tol_value = pd.concat(summer_aae_tol_hour, axis = 1)
    summer_aae_tol_hour = []
    summer_aae_tol_year.append(summer_aae_tol_value)

diurnal_aae_tol_summer = pd.concat(summer_aae_tol_year)
diurnal_aae_tol_column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
diurnal_aae_tol_summer.columns = diurnal_aae_tol_column
diurnal_aae_tol_summer = diurnal_aae_tol_summer.reset_index(drop = True)
diurnal_aae_tol_summer.index = diurnal_aae_tol_row
diurnal_aae_tol_summer = diurnal_aae_tol_summer.T
diurnal_aae_tol_summer = diurnal_aae_tol_summer.dropna(axis = 1)
diurnal_aae_tol_row = []
diurnal_aae_tol_row = diurnal_aae_tol_summer.columns

diurnal_aae_tol_summer.plot(marker = 's')
plt.title('PM$_1$')
plt.ylabel('AAE') 
plt.xlabel('Hour')
plt.legend(diurnal_aae_tol_row, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()

plt.title('PM$_1$') 
diurnal_aae_tol_summer.boxplot(showfliers = False, color = 'tab:red', boxprops = {'linewidth': 2, 'color': 'tab:red'}, medianprops = {'linewidth': 2, 'color': 'tab:red'}, whiskerprops = {'linewidth': 2, 'color': 'tab:red'}, capprops = {'linewidth': 2, 'color': 'tab:red'}, whis = [10, 90], rot = 90) 
plt.xlabel('Year)')
plt.ylabel('AAE')
plt.show()

pm_1_tol_autumn = pm_1_tol_autumn.dropna()

wavelengths = [370, 470, 520, 590, 660, 880, 950]
x = pd.DataFrame(np.log(wavelengths))
x = x.dropna()
x = x.replace([np.inf, -np.inf], np.nan).dropna()
x = x.transpose()
y = pd.DataFrame(np.log(pm_1_tol_autumn.iloc[:, 1:8]))
y = y.dropna()
y = y.replace([np.inf, -np.inf], np.nan).dropna()
pm_1_tol_autumn = pm_1_tol_autumn.loc[y.index]
y = y.transpose()
model = LinearRegression()
slope = pd.DataFrame({'slope': [(model.fit(x.values.reshape(-1, 1), y.iloc[:, idx]).coef_[0]) for idx in range(y.shape[1])]})
tol_autumn_subset_1 = pm_1_tol_autumn.iloc[:, :15]
tol_autumn_subset_1 = tol_autumn_subset_1.reset_index()
tol_autumn_subset_1 = tol_autumn_subset_1.iloc[:, 1:]
slope = slope.reset_index()
slope = slope.iloc[:, 1]
slope = - slope
tol_autumn_subset_2 = pm_1_tol_autumn.iloc[:, 15]
tol_autumn_subset_2 = tol_autumn_subset_2.reset_index()
tol_autumn_subset_2 = tol_autumn_subset_2.iloc[:, 1]
tol_autumn_list = []
tol_autumn_list.append(tol_autumn_subset_1)
tol_autumn_list.append(slope)
tol_autumn_list.append(tol_autumn_subset_2)
pm_1_tol_autumn = pd.concat(tol_autumn_list, axis = 1) 

autumn_aae_tol_hour = []
autumn_aae_tol_year = []
start_year = pm_1_tol_autumn.iloc[0, 0].year
end_year_plus_one = pm_1_tol_autumn.iloc[-1, 0].year + 1
time_step = 1

diurnal_aae_tol_row = []

for i in range (start_year, end_year_plus_one):
    diurnal_aae_tol_row.append(i)
    
diurnal_aae_tol_autumn = []

start_year = pm_1_tol_autumn.iloc[0, 0].year
end_year_plus_one = pm_1_tol_autumn.iloc[-1, 0].year + 1
time_step = 1

start_ah = pm_1_tol_autumn.iloc[:, -1].min()
end_ah_plus_one = pm_1_tol_autumn.iloc[:, -1].max() + 1
time_ah = 1

for i in range(start_year, end_year_plus_one, time_step):
    for j in range(int(start_ah), int(end_ah_plus_one), int(time_ah)):
        diurnal_aae_tol_sep = pm_1_tol_autumn[(pm_1_tol_autumn.iloc[:, 0].dt.year == i) & (pm_1_tol_autumn.iloc[:, 0].dt.month == 9) & (pm_1_tol_autumn.iloc[:, -1] == j)]
        diurnal_aae_tol_oct = pm_1_tol_autumn[(pm_1_tol_autumn.iloc[:, 0].dt.year == i) & (pm_1_tol_autumn.iloc[:, 0].dt.month == 10) & (pm_1_tol_autumn.iloc[:, -1] == j)]
        diurnal_aae_tol_nov = pm_1_tol_autumn[(pm_1_tol_autumn.iloc[:, 0].dt.year == i) & (pm_1_tol_autumn.iloc[:, 0].dt.month == 11) & (pm_1_tol_autumn.iloc[:, -1] == j)]
        diurnal_aae_tol_autumn.append(diurnal_aae_tol_sep)
        diurnal_aae_tol_autumn.append(diurnal_aae_tol_oct)
        diurnal_aae_tol_autumn.append(diurnal_aae_tol_nov)
        diurnal_aae_tol_row = pd.concat(diurnal_aae_tol_autumn)
        diurnal_aae_tol_autumn = []
        autumn_aae_tol_value = diurnal_aae_tol_row.iloc[:, -2]
        autumn_aae_tol_value = autumn_aae_tol_value.to_frame()
        autumn_aae_tol_hour.append(autumn_aae_tol_value)
    autumn_aae_tol_value = pd.concat(autumn_aae_tol_hour, axis = 1)
    autumn_aae_tol_hour = []
    autumn_aae_tol_year.append(autumn_aae_tol_value)
    
diurnal_tol_autumn_row = pd.concat(autumn_aae_tol_year)

diurnal_aae_tol_column = []

for i in range(int(start_ah), int(end_ah_plus_one), int(time_ah)):
    diurnal_aae_tol_column.append(i)
    
diurnal_tol_autumn_row.columns = diurnal_aae_tol_column

tol_hour = 90
tol_5_h_group = tol_hour // 5

group_5_h_index = [range(i*5, (i + 1)*5) for i in range(tol_5_h_group)]

diurnal_tol_autumn_average = pd.concat([diurnal_tol_autumn_row.iloc[:, indices].mean(axis = 1) for indices in group_5_h_index], axis = 1)

old_column_name = diurnal_tol_autumn_row.columns
new_column_name = [str(int((int(old_column_name[i]) + int(old_column_name[i + 4])) / 2)) for i in range(0, tol_hour, 5)]
diurnal_tol_autumn_average.columns = new_column_name

plt.title('PM$_1$') 
diurnal_tol_autumn_average.boxplot(showfliers = False, color = 'tab:orange', boxprops = {'linewidth': 2, 'color': 'tab:orange'}, medianprops = {'linewidth': 2, 'color': 'tab:orange'}, whiskerprops = {'linewidth': 2, 'color': 'tab:orange'}, capprops = {'linewidth': 2, 'color': 'tab:orange'}, whis = [10, 90], rot = 90) 
plt.xlabel('Time over land (hour)')
plt.ylabel('AAE')
plt.legend(['Autumn season'], bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()

autumn_aae_tol_hour = []
autumn_aae_tol_year = []
start_year  = pm_1_tol_autumn.iloc[0, 0].year
end_year_plus_one = pm_1_tol_autumn.iloc[-1, 0].year + 1
time_step = 1

diurnal_aae_tol_row = []

for i in range (start_year, end_year_plus_one, time_step):
     diurnal_aae_tol_row.append(i)

for i in range(start_year, end_year_plus_one, time_step):
    for j in range(0, 24, 1):
        diurnal_aae_tol_autumn = pm_1_tol_autumn[(pm_1_tol_autumn.iloc[:, 0].dt.year == i) & (pm_1_tol_autumn.iloc[:, 0].dt.month == 9) & (pm_1_tol_autumn.iloc[:, 0].dt.hour == j) | (pm_1_tol_autumn.iloc[:, 0].dt.year == i) & (pm_1_tol_autumn.iloc[:, 0].dt.month == 10) & (pm_1_tol_autumn.iloc[:, 0].dt.hour == j) | (pm_1_tol_autumn.iloc[:, 0].dt.year == i) & (pm_1_tol_autumn.iloc[:, 0].dt.month == 11) & (pm_1_tol_autumn.iloc[:, 0].dt.hour == j)]
        autumn_aae_tol_value = diurnal_aae_tol_autumn.iloc[:, -2]
        autumn_aae_tol_value = autumn_aae_tol_value.to_frame()
        autumn_aae_tol_value = autumn_aae_tol_value.median()
        autumn_aae_tol_hour.append(autumn_aae_tol_value)
    autumn_aae_tol_value = pd.concat(autumn_aae_tol_hour, axis = 1)
    autumn_aae_tol_hour = []
    autumn_aae_tol_year.append(autumn_aae_tol_value)

diurnal_aae_tol_autumn = pd.concat(autumn_aae_tol_year)
diurnal_aae_tol_column = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
diurnal_aae_tol_autumn.columns = diurnal_aae_tol_column
diurnal_aae_tol_autumn = diurnal_aae_tol_autumn.reset_index(drop = True)
diurnal_aae_tol_autumn.index = diurnal_aae_tol_row
diurnal_aae_tol_autumn = diurnal_aae_tol_autumn.T
diurnal_aae_tol_autumn = diurnal_aae_tol_autumn.dropna(axis = 1)
diurnal_aae_tol_row = []
diurnal_aae_tol_row = diurnal_aae_tol_autumn.columns

diurnal_aae_tol_autumn.plot(marker = 's')
plt.title('PM$_1$')
plt.ylabel('AAE') 
plt.xlabel('Hour')
plt.legend(diurnal_aae_tol_row, bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()

plt.title('PM$_1$') 
diurnal_aae_tol_autumn.boxplot(showfliers = False, color = 'tab:orange', boxprops = {'linewidth': 2, 'color': 'tab:orange'}, medianprops = {'linewidth': 2, 'color': 'tab:orange'}, whiskerprops = {'linewidth': 2, 'color': 'tab:orange'}, capprops = {'linewidth': 2, 'color': 'tab:orange'}, whis = [10, 90], rot = 90) 
plt.xlabel('Year)')
plt.ylabel('AAE')
plt.show()














