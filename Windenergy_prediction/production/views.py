from django.shortcuts import render,redirect, HttpResponse
from django.contrib.auth.models import auth, User
import folium
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mpld3
from pandas_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
from .models import wind_details,location_details,file_details
from tensorflow.keras.models import load_model
from django.views.decorators.cache import cache_control

pred_list = []

#Create your views here.
def main(request):
    username = User.objects.get(username=request.user.username)
    return render(request, "user_home.html", {"username": username})
def direct(request):
    username = User.objects.get(username=request.user.username)
    return render(request, "My-home.html", {"username": username})

def map(request):
    return render(request,"map.html")
def map_display(request):
    return render(request, "map.html")

def add_location_map(request):
    username = User.objects.get(username=request.user.username)
    lat = request.POST['lot']
    lon = request.POST['log']
    machine_name = request.POST['mn']
    ld = location_details()
    ld.user_name = username
    ld.lattitude = lat
    ld.longtitude = lon
    ld.machine_name = machine_name
    ld.save()
    loc_info = location_details.objects.filter(user_name= username).values('lattitude','longtitude','machine_name')
    latt = []
    longi = []
    mn = []
    for i in loc_info:
        latt.append(i['lattitude'])
        longi.append(i['longtitude'])
        mn.append(i['machine_name'])

    loc_info1 = wind_details.objects.filter(username=username).values('latitude', 'longitude', 'orgname')
    for i in loc_info1:
        latt.append(i['latitude'])
        longi.append(i['longitude'])
        mn.append(i['orgname'])
    #map = folium.Map(location=[lat,lon], zoom_start=6,tiles="Stamen Terrain")
    map = folium.Map()
    fg = folium.FeatureGroup(name="test")
    for i, j, k in zip(latt, longi, mn):
        fg.add_child(folium.Marker(location=[i,j], popup=k, icon=folium.Icon(color='green')))
    map.add_child(fg)
    map.save("Templates/map.html")
    return render(request, "Wind farm Location.html", {"username": username})

def update_profile(request):
    un = User.objects.get(username=request.user.username)
    fn = request.POST['fname']
    ln = request.POST['lname']
    phone = request.POST['phone']
    pincode = request.POST['pincode']
    location = request.POST['location']
    orgname = request.POST['orgname']
    orgemail = request.POST['orgemail']
    country = request.POST['country']
    state = request.POST['state']
    no_of_wind = request.POST['no_of_wind']
    latt = request.POST['latt']
    long = request.POST['long']
    wd = wind_details()
    wd.username = un
    wd.first_name = fn
    wd.last_name = ln
    wd.phone = phone
    wd.pincode = pincode
    wd.location = location
    wd.orgname = orgname
    wd.orgemail = orgemail
    wd.country = country
    wd.state = state
    wd.no_of_wind = no_of_wind
    wd.latitude = latt
    wd.longitude = long
    wd.save()

    loc_info1 = wind_details.objects.filter(username=un).values('latitude', 'longitude', 'orgname')
    latt = []
    long = []
    mn = []
    for i in loc_info1:
        latt.append(i['latitude'])
        long.append(i['longitude'])
        mn.append(i['orgname'])

    loc_info = location_details.objects.filter(user_name=un).values('lattitude', 'longtitude', 'machine_name')
    for i in loc_info:
        latt.append(i['lattitude'])
        long.append(i['longtitude'])
        mn.append(i['machine_name'])


    # map = folium.Map(location=[lat,lon], zoom_start=6,tiles="Stamen Terrain")
    map = folium.Map()
    fg = folium.FeatureGroup(name="test")
    for i, j, k in zip(latt, long, mn):
        fg.add_child(folium.Marker(location=[i, j], popup=k, icon=folium.Icon(color='green')))
    map.add_child(fg)
    map.save("Templates/map.html")
    return render(request, "My-home.html")

def comp_analysis(request):
    return render(request, "comparative analysis.html")

def windform_location(request):
    return render(request, "Wind farm Location.html")

def logout(request):
    auth.logout(request)
    return redirect('/')

def handle_uploaded_file(f):
    with open('production/data.xls', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def upload(request):
    if request.method == 'POST':
        f = file_details(request.POST, request.FILES)
        if f.is_valid():
            handle_uploaded_file(request.FILES['file'])
            return render(request, "comparative analysis.html", {'result': 'file successfully Uploaded'})
        else:
            return render(request, "comparative analysis.html", {'result': 'error while uploading file'})

def show_analysis(request):
    df = pd.read_excel('production/data.xls')
    df.to_csv('data.csv', encoding='latin-1')
    prof = ProfileReport(df, minimal=True)
    prof.to_file(output_file='Templates/analysis_output.html')
    return render(request, "analysis_output.html")

def show_graph(request):
    df = pd.read_excel('production/data.xls')
    df.to_csv('production/data.csv')
    df = pd.read_csv('production/data.csv')
    print(df.dtypes)
    column_name = request.POST['column_name']
    pred_range = int(request.POST['slider'])
    df1 = df.reset_index()[column_name]

    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

    training_size = int(len(df1) * 0.65)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

    import numpy
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    #Model Development

    model = load_model('production/model.h5')

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    fig = plt.figure()
    look_back = 100
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict
    # plot baseline and predictions
    # plt.plot(scaler.inverse_transform(df1))
    # plt.plot(trainPredictPlot)
    # plt.plot(testPredictPlot)
    # plt.show()
    #fig.savefig("production/Algorithm_working.png")


    x_input = test_data[len(test_data)-100:].reshape(1, -1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = 100
    i = 0
    while (i < pred_range):
        if (len(temp_input) > 100):
            # print(temp_input)
            x_input = np.array(temp_input[1:])
            print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            # print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            # print(temp_input)
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i + 1

    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 101 + pred_range)

    pred_list = lst_output
    # fig = plt.figure()
    # plt.plot(day_new, scaler.inverse_transform(df1[len(df1) - 100:]))
    # plt.plot(day_pred, scaler.inverse_transform(lst_output))
    #fig.savefig("Analysis_Images/model_prediction.png")

    plt.rcParams["figure.figsize"] = (13, 6)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.plot(scaler.inverse_transform(df1))
    ax1.plot(trainPredictPlot)
    ax1.plot(testPredictPlot)
    ax1.set_xlabel("Number of Days")
    ax1.set_ylabel("Amount of Power Produced")
    ax1.set_title("Algorithm Performance on Existing Data")

    ax2.plot(day_new, scaler.inverse_transform(df1[len(df1) - 100:]))
    ax2.plot(day_pred, scaler.inverse_transform(lst_output))
    ax2.set_xlabel("Number of Days")
    ax2.set_ylabel("Amount of Power Produced")
    ax2.set_title("Algorithm Performance on Existing Data")

    plt.show()
    mpld3.save_html(fig, "Templates/prediction_graph.html")

    return render(request, "prediction_graph.html")

# def show_predict_values(request):
#     return render(request, "prediction_output.html",{'output' : pred_list})
