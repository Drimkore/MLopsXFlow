from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

args = {
    'owner': 'admin',
    'start_date': dt.datetime(2023, 12, 1),
    'retries': 1,
    'retry_delays': dt.timedelta(minutes=1),
    'depends_on_past': False,
    'provide_context': True
}

with DAG(
    dag_id='gpstore',
    default_args=args,
    schedule_interval=None,
    tags=['google', 'score'],
) as dag:
    get_data = BashOperator(task_id='get_data',
                            bash_command="cd /home/art/gitprj/MLopsXFlow/datasets | gdown 1i1W2wtpQErLjJdbVVThsqFd_oE2BDspj", 
                            dag=dag)
    extract_data = BashOperator(task_id='extract_data',
                            bash_command="unzip -o -q archive.zip -d ./data/raw/", 
                            dag=dag)
    data_one = BashOperator(task_id='data_one',
                            bash_command="python /home/art/gitprj/MLopsXFlow/scripts/data_onep.py", 
                            dag=dag)
    data_second = BashOperator(task_id='data_second',
                            bash_command="python /home/art/gitprj/MLopsXFlow/scripts/data_second.py", 
                            dag=dag)
    data_third = BashOperator(task_id='data_third',
                            bash_command="python /home/art/gitprj/MLopsXFlow/scripts/data_third.py", 
                            dag=dag)
    data_fourth = BashOperator(task_id='data_fourth',
                            bash_command="python /home/art/gitprj/MLopsXFlow/scripts/data_fourth.py", 
                            dag=dag)
    data_fifth = BashOperator(task_id='data_fifth',
                            bash_command="python /home/art/gitprj/MLopsXFlow/scripts/data_fifth.py", 
                            dag=dag)
    data_six = BashOperator(task_id='data_six',
                            bash_command="python /home/art/gitprj/MLopsXFlow/scripts/data_six.py", 
                            dag=dag)

    get_data >> extract_data >> data_one >> data_second >> data_third >> data_fourth >> data_fifth >> data_six


