# Generated by Django 2.1.2 on 2018-10-21 07:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0005_auto_20181021_0337'),
    ]

    operations = [
        migrations.AddField(
            model_name='filter',
            name='smooth',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='filter',
            name='state',
            field=models.IntegerField(default=1),
        ),
    ]
