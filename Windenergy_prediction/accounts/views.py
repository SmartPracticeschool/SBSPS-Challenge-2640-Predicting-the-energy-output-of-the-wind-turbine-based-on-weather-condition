from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth.models import auth,User
from django.contrib.auth import authenticate
from .models import user_details

from production.models import wind_details

# Create your views here.
def login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            auth.login(request,user)
            username = User.objects.get(username=request.user.username)
            usr_detail = wind_details.objects.filter(username=username).values()
            if len(usr_detail) == 0:
                return redirect('production/main')
            else:
                return redirect('production/direct')
        else:
            return render(request, 'login.htm', {'error': 'Wrong Credentials'})
    else:
        return render(request, 'login.htm')


def register(request):
    if request.method == 'POST':
        name = request.POST['name']
        email = request.POST['email']
        username = request.POST['username']
        pass1 = request.POST['password']
        pass2 = request.POST['confirm']
        if pass1 != pass2:
            return render(request,"registration.htm",{"error": "password did not match"})
        if User.objects.filter(username=username).exists():
            return render(request, 'registration.htm', {'error': 'Username already taken'})
        elif User.objects.filter(email=email).exists():
            return render(request, 'registration.htm', {'error': 'email already taken'})
        else:
            user = User.objects.create_user(username=username, email=email,
                                            password=pass1)
        user.save()
        ud = user_details()
        ud.Name = name
        ud.email = email
        ud.username = username
        ud.save()
        return render(request, "login.htm", {'error': 'User Created Successfully'})
    else:
        return render(request, "registration.htm")