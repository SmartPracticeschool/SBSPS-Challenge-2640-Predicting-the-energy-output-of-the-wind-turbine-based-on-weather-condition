from django.db import models
from django import forms
# Create your models here.
class wind_details(models.Model):
    username = models.CharField(max_length=100)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    phone = models.CharField(max_length=100)
    pincode = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    orgname = models.CharField(max_length=100)
    orgemail = models.CharField(max_length=100)
    country = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    no_of_wind = models.CharField(max_length=100)

class location_details(models.Model):
    user_name = models.CharField(max_length=100)
    lattitude = models.CharField(max_length=100)
    longtitude = models.CharField(max_length=100)
    machine_name = models.CharField(max_length=100)

class file_details(forms.Form):
    file = forms.FileField()
