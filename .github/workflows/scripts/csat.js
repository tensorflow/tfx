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
const CONSTANT_VALUES = require('./constant');

/**
 * Invoked from stale_csat.js and csat.yaml file to post survey link
 * in closed issue.
 * @param {!Object.<string,!Object>} github contains pre defined functions.
 *  context Information about the workflow run.
 * @return {null}
 */
module.exports = async ({ github, context }) => {
    const issue = context.payload.issue.html_url;
    let baseUrl = '';
    // Loop over all ths label present in issue and check if specific label is
    // present for survey link.
    for (const label of context.payload.issue.labels) {
        if (label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.BUG) ||
            label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.BUG_INSTALL) ||
            label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.TYPE_PERFORMANCE) ||
            label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.TYPE_OTHER) ||
            label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.TYPE_SUPPORT) ||
            label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.TYPE_DOCS_BUG)) {
            console.log(
                `label-${label.name}, posting CSAT survey for issue =${issue}`);
            if (context.repo.repo.includes('mediapipe'))
                baseUrl = CONSTANT_VALUES.MODULE.CSAT.MEDIA_PIPE_BASE_URL;
            else
                baseUrl = CONSTANT_VALUES.MODULE.CSAT.BASE_URL;

            const yesCsat = `<a href="${baseUrl + CONSTANT_VALUES.MODULE.CSAT.SATISFACTION_PARAM +
                CONSTANT_VALUES.MODULE.CSAT.YES +
                CONSTANT_VALUES.MODULE.CSAT.ISSUEID_PARAM + issue}"> ${CONSTANT_VALUES.MODULE.CSAT.YES}</a>`;

            const noCsat = `<a href="${baseUrl + CONSTANT_VALUES.MODULE.CSAT.SATISFACTION_PARAM +
                CONSTANT_VALUES.MODULE.CSAT.NO +
                CONSTANT_VALUES.MODULE.CSAT.ISSUEID_PARAM + issue}"> ${CONSTANT_VALUES.MODULE.CSAT.NO}</a>`;
            const comment = CONSTANT_VALUES.MODULE.CSAT.MSG + '\n' + yesCsat + '\n' +
                noCsat + '\n';
            let issueNumber = context.issue.number ?? context.payload.issue.number;
            await github.rest.issues.createComment({
                issue_number: issueNumber,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: comment
            });
        }
    }
};