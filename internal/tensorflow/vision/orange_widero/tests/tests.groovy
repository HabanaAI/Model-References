import java.text.SimpleDateFormat

def set_aws_authentication()
{
    def security = new mobileye.devops.services.security()
    (AWS_SECRET, AWS_ACCESS) = security.GetFileSecret("2ebd1c1c-b642-4319-9970-9181639ad3aa")
    env.AWS_ACCESS_KEY_ID = AWS_ACCESS
    env.AWS_SECRET_ACCESS_KEY = AWS_SECRET
    env.AWS_DEFAULT_REGION = 'us-east-1'
}


def run_tests()
{
    set_aws_authentication()
    def date = new Date()
    def sdf = new SimpleDateFormat("yyyyMMddHHmmss")
    def datetime_str = sdf.format(date)
    def interpreter = "/mobileye/algo_STEREO3/stereo/venv/latest/bin/python"
    def base_reports_path = "/mobileye/algo_STEREO3/stereo/tests/reports"
    def report_path = String.format("%s/%s", base_reports_path, datetime_str)
    def command = "%s -m pytest -srA tests/ --local --html=%s/tests_report.html"
    command += " --cov=stereo --cov-config=tests/setup.cfg --cov-report html:%s/coverage_report"
    def full_command = String.format(command, interpreter, report_path, report_path)
    println "Creating report outdir"
    sh String.format("mkdir %s", report_path)
    println "running tests"
    sh full_command
    return report_path
}

def run_tests_after_commit()
{
    set_aws_authentication()
    def sdf = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss")
    def last_commit_timestamp = sh(returnStdout: true, script: "git log -1 --format=%at").trim().toLong() * 1000
    def last_commit_date = new Date(last_commit_timestamp)
    println String.format("last commit time: %s", sdf.format(last_commit_date))

    def now = new Date()
    def yesterday = now - 1
    def last_build_time = yesterday.getTime()
    println String.format("last build time: %s", sdf.format(yesterday))

    if (last_build_time < last_commit_timestamp)
    {
        println "New push, running daily tests"
        return run_tests()
    }
    else
    {
        println "Nothing was pushed into master for the last 24 hours, no need to run tests"
        return 0
    }
}

return this