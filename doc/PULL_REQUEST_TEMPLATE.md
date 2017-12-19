# # Pull Request Checklist

 * Merge in the latest Develop branch changes to your branch
 * Remove .pyc files from your repository: 
  * Linux/Mac
     * find . -name \*.pyc -delete
  * Windows
     * del /S *.pyc
 * Run automatic regression and makes sure everything is passing
 * Did you make a new regression test that covers your new code?
 * Did you update your docstrings?
 * Did you update your headers to include your name and date?
 * Do a final compare with the Develop branch to be sure what you're changing
