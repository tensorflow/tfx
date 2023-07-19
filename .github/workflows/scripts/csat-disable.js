/*
Copyright 2023 Google LLC. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
const CONSTENT_VALUES = require('./constant');

/**
 * Invoked from csat-disable.yaml pick last 3 frames of issues and
 * disable the all csat-survey links in closed issue.
 * @param {!Object.<string,!Object>} github contains pre defined functions.
 *  context Information about the workflow run.
 */
module.exports = async ({github, context}) => {
  
  console.log('Owner of the repo = ', context.payload.repository.owner.login);
  for (let i = 1; i < 4; i++) {
    console.log('Running for page :', i);
    const issueDetails = await github.rest.issues.listForRepo({
      owner: context.payload.repository.owner.login,
      repo: context.payload.repository.name,
      per_page: 100,
      sort: 'updated',
      state: 'closed',
      page: i,
    });
    const listIssues = issueDetails.data;
    for (let i = 0; i < listIssues.length; i++) {
      const issue = listIssues[i];
      console.log('Pick issue number : ', issue.number);
      console.log('Issue closed at : ', issue.closed_at);
      let issueClose = new Date(issue.closed_at);
      let currentEpoch = new Date();
      let timeDiff = currentEpoch - issueClose;
      let diffInDays = timeDiff / (1000 * 60 * 60 * 24);
      console.log(`Issue is ${diffInDays} days old.`);
      
      if (diffInDays <= 7) continue;
      console.log(
          'Fetching all the comments from issue number :', issue.number);

      const issueComments = await github.rest.issues.listComments({
        owner: context.payload.repository.owner.login,
        repo: context.payload.repository.name,
        per_page: 100,
        issue_number: issue.number,
      });

      const commentsList = issueComments.data;
      for (let i = 0; i < commentsList.length; i++) {
        let comment = commentsList[i];
        let commentTime = new Date(comment.created_at);
        let currentEpoch = new Date();
        let timeDiff = currentEpoch - commentTime;
        let diffInDays = timeDiff / (1000 * 60 * 60 * 24);
        if (diffInDays >= 7 && comment.created_at &&
            comment.body.indexOf(CONSTENT_VALUES.MODULE.CSAT.MSG) != -1) {
          console.log('CSAT link disabled for issue number: ', issue.number);
          await github.rest.issues.updateComment({
            owner: context.payload.repository.owner.login,
            repo: context.payload.repository.name,
            comment_id: comment.id,
            body: CONSTENT_VALUES.MODULE.CSAT.DISABLEMSG + '\n' +
                'Yes' +
                '\n' +
                'No' +
                '\n'
          });
        }
      }
    }
  }
};