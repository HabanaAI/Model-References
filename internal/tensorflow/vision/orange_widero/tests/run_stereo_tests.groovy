@Library('Jenkins_Shared_Lib@1.1.x')
def run_tests()
{
    def tests = load "tests/tests.groovy"
    tests.run_tests()
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
        stage('run_tests')
        {
            steps
            {
                run_tests()
            }
        }
    }
}