# Generated by Django 3.0.7 on 2020-06-25 14:12

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='wind_details',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('first_name', models.CharField(max_length=100)),
                ('last_name', models.CharField(max_length=100)),
                ('phone', models.CharField(max_length=100)),
                ('pincode', models.CharField(max_length=100)),
                ('location', models.CharField(max_length=100)),
                ('orgname', models.CharField(max_length=100)),
                ('orgemail', models.CharField(max_length=100)),
                ('country', models.CharField(max_length=100)),
                ('state', models.CharField(max_length=100)),
                ('no_of_wind', models.CharField(max_length=100)),
            ],
        ),
    ]
