import sys
import os
from datetime import datetime, timedelta
import boto3


LOG_GROUP = "/aws/sagemaker/TrainingJobs"


def event_to_str(event):
    event_str = f"{datetime.fromtimestamp(event['timestamp']*1e-3)}\t{event['message']}"
    return event_str


def retrieve_all_results(aws_func, aggregate_key, kwargs):
    results = []
    search_completed = False
    counter = 0
    while not search_completed:
        counter += 1
        response = aws_func(**kwargs)
        # print(f"Query {counter} | Number of results: {len(response[aggregate_key])}")
        results.extend(response[aggregate_key])
        if 'nextToken' in response:
            kwargs.update(nextToken=response['nextToken'])
        elif 'NextToken' in response:
            kwargs.update(NextToken=response['NextToken'])
        else:
            search_completed = True
    print(f"Number of results: {len(results)}")
    return results


def get_cw_logs(args):
    client = boto3.client('logs')

    streams = client.describe_log_streams(
        logGroupName=LOG_GROUP,
        logStreamNamePrefix=args.job_name
    )
    stream_name = [s['logStreamName'] for s in streams['logStreams']
                   if "platform-logs" not in s['logStreamName']][0]

    log_events_kwargs = dict(
        logGroupName=LOG_GROUP,
        logStreamNames=[
            stream_name
        ]
    )
    if args.hours is not None:
        start_time = int((datetime.now() - timedelta(hours=args.hours)).timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        log_events_kwargs.update(startTime=start_time, endTime=end_time)
    if args.filter_pattern is not None:
        log_events_kwargs.update(filterPattern=args.filter_pattern)

    events = retrieve_all_results(client.filter_log_events,
                                  kwargs=log_events_kwargs,
                                  aggregate_key='events')
    return events


def get_sm_jobs(args):
    client = boto3.client('sagemaker')
    name = args.name
    if name is None:
        name = os.environ['USER']
        print(f"Using default search name: {name}")

    filters = [
        {'Name': 'TrainingJobName',
         'Operator': 'Contains',
         'Value': name}
    ]
    if args.in_progress:
        filters.append(
            {'Name': 'TrainingJobStatus',
             'Operator': 'Equals',
             'Value': 'InProgress'}
        )

    search_kwargs = dict(
        Resource='TrainingJob',
        SearchExpression={'Filters': filters}
    )

    jobs = retrieve_all_results(client.search,
                                kwargs=search_kwargs,
                                aggregate_key='Results')
    return jobs


def aws_logs_main(args):
    events = get_cw_logs(args)
    event_lines = os.linesep.join([event_to_str(e) for e in events]) + os.linesep
    if args.out is not None:
        with open(args.out, 'w') as f:
            f.write(event_lines)
    else:
        sys.stdout.write(event_lines)


def aws_jobs_main(args):
    jobs = get_sm_jobs(args)
    jobs = sorted(jobs, key=lambda r: r['TrainingJob']['LastModifiedTime'], reverse=True)
    jobs_lines = []

    limit = min(args.limit, len(jobs))
    print(f"Showing first {limit} jobs:")
    for i in range(limit):
        job_name = jobs[i]['TrainingJob']['TrainingJobName']
        job_status = jobs[i]['TrainingJob']['TrainingJobStatus']
        start_time = jobs[i]['TrainingJob'].get('TrainingStartTime', None)
        end_time = jobs[i]['TrainingJob'].get('TrainingEndTime', None)
        jobs_lines.append(f"{job_status:11} | {job_name:68} | {start_time} | {end_time}")
    jobs_lines = os.linesep.join(jobs_lines) + os.linesep
    print(jobs_lines)
    # sys.stdout.write()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Retrieve CloudWatch logs and SageMaker jobs from AWS")
    subparsers = parser.add_subparsers(dest="mode")
    subparsers.required = True

    logs_parser = subparsers.add_parser('logs', help='Retrieve Cloud Watch logs')
    logs_parser.add_argument('job_name', help="AWS Job name")
    logs_parser.add_argument('-H', '--hours', type=float,
                        help="Retrieve logs from the last given hours")
    logs_parser.add_argument('-f', '--filter_pattern', type=str,
                        help="Filter pattern. If not specified, all events are returned")
    logs_parser.add_argument('-o', '--out', type=str,
                        help="Output file. If not specified, events will be printed to stdout")

    jobs_parser = subparsers.add_parser('jobs', help='Retrieve SageMaker jobs')
    jobs_parser.add_argument('-n', '--name',
                             help="Search for this string in training jobs name. "
                                  "If not specified, search for jobs which contain the username.")
    jobs_parser.add_argument('-l', '--limit', type=int, default=10,
                             help="The maximum number of jobs to return (newest to oldest). default: %(default)s")
    jobs_parser.add_argument('-p', '--in_progress', action='store_true',
                             help="Show only jobs in progress (do not show failed/completed/stopped jobs")

    args = parser.parse_args()

    if args.mode == 'logs':
        aws_logs_main(args)
    elif args.mode == 'jobs':
        aws_jobs_main(args)
    else:
        raise ValueError()

