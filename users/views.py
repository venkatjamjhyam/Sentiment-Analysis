from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from .AlgoProcess.modelTraining import Algorithms

algo = Algorithms()



# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account has not been activated by the Admin🛑🤚')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def view_data(request):
    from django.conf import settings
    import pandas as pd
    import numpy as np
    path = settings.MEDIA_ROOT + '\\' + 'Twitter_Data.csv'
    data = pd.read_csv(path)
    data.reset_index()
    data = data.to_html()
    return render(request,'users/view.html',{'data':data})



def UserTraining(request):

    loss,acc = algo.RNN()
    loss1,acc1 = algo.LSTM()
    cr_log = algo.LogisticRegression()
    cr_nv = algo.GaussinNB()
    cr_dc = algo.DecisionTree()

    return render(request,'users/UserTraining.html' ,{'acc':acc,'loss':loss,'loss1':loss1,'acc1':acc1,
                                                      'cr_log':cr_log,'cr_nv':cr_nv,'cr_dc':cr_dc})

def predict(request):
    if request.method=='POST':
        
        joninfo  = request.POST.get('joninfo')
        result = algo.predict(joninfo)
        print(result)
        return render(request, 'users/testform.html', {'result': result})
    else:
        return render(request,'users/testform.html',{})



# def prediction(request):
#     if request.method=='POST':
#         from .utility.resnet_50 import predict
#         joninfo  = request.POST.get('joninfo')
#         result = predict(joninfo)
#         print(request)
#         return render(request, 'users/testform.html', {'result': result})
#     else:
#         return render(request,'users/testform.html',{})