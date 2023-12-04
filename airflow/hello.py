from airflow import DAG
import airflow
from datetime import datetime
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator

def test123():
    print("Hello world!")

args = {
    'owner': 'Artem Bezzubov',
    'start_date':datetime(2023, 11, 1),
    'provide_context':True
}

with DAG('hello', description='hello', schedule='*/1 * * * *',  catchup=False, default_args=args) as dag: 
    task_1 = PythonOperator(task_id="task_1", python_callable=test123)


