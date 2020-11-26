@Library('Jenkins_Shared_Lib@1.1.x')
def run_tests_after_commit()
{
    def tests = load "tests/tests.groovy"
    tests.run_tests_after_commit()
}
pipeline
{
    agent
    {
        node
        {
            label 'vidar-deb8-x86_64'
        }
    }
    stages
    {
        stage('run_tests_after_commit')
        {
            steps
            {
                script
                {
                    report_path = run_tests_after_commit()
                }
            }
        }
        stage('send an email')
        {

            when{ not {equals expected: 0, actual: report_path}}
            steps
            {
                emailext body: "The tests last run result is - ${currentBuild.currentResult}. report could be found at - file://${report_path}/tests_report.html",
                subject: "Daily stereo tests status - ${currentBuild.currentResult}",
                recipientProviders: [[$class: 'DevelopersRecipientProvider']]
            }
        }
    }
}