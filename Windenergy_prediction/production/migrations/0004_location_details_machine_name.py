# Generated by Django 3.0.7 on 2020-06-26 13:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('production', '0003_location_details'),
    ]

    operations = [
        migrations.AddField(
            model_name='location_details',
            name='machine_name',
            field=models.CharField(default=0, max_length=100),
            preserve_default=False,
        ),
    ]
