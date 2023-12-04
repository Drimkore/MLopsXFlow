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
                            bash_command="cd /home/art/gitprj/MLopsXFlow/datasets && gdown 1i1W2wtpQErLjJdbVVThsqFd_oE2BDspj", 
                            dag=dag)
    extract_data = BashOperator(task_id='extract_data',
                            bash_command="mkdir data && cd data && mkdir raw && unzip -o -q /home/art/gitprj/MLopsXFlow/datasets/archive.zip -d /home/art/gitprj/MLopsXFlow/datasets/data/raw/", 
                            dag=dag)
    data_prep_one = BashOperator(task_id='data_prep_one',
                            bash_command="python /home/art/gitprj/MLopsXFlow/scripts/data_prep_one.py", 
                            dag=dag)
    data_prep_second = BashOperator(task_id='data_prep_second',
                            bash_command="python /home/art/gitprj/MLopsXFlow/scripts/data_prep_second.py", 
                            dag=dag)
    data_prep_third = BashOperator(task_id='data_prep_third',
                            bash_command="python /home/art/gitprj/MLopsXFlow/scripts/data_prep_third.py", 
                            dag=dag)
    data_split_fourth = BashOperator(task_id='data_split_fourth',
                            bash_command="cd /home/art/gitprj/MLopsXFlow/ && python /home/art/gitprj/MLopsXFlow/scripts/data_split_fourth.py", 
                            dag=dag)
    data_train_fifth = BashOperator(task_id='data_train_fifth',
                            bash_command="cd /home/art/gitprj/MLopsXFlow/ && python /home/art/gitprj/MLopsXFlow/scripts/data_train_fifth.py", 
                            dag=dag)
    data_test_six = BashOperator(task_id='data_test_six',
                            bash_command="cd /home/art/gitprj/MLopsXFlow/ && python /home/art/gitprj/MLopsXFlow/scripts/data_test_six.py", 
                            dag=dag)

    get_data >> extract_data >> data_prep_one >> data_prep_second >> data_prep_third >> data_split_fourth >> data_train_fifth >> data_test_six


