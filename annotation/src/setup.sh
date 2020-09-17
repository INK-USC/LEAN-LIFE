./create_empty_db.sh $1
python manage.py migrate
python manage.py createsuperuser