# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import unittest

import requests
from requests.exceptions import HTTPError

from transformers.hf_api import HfApi, HfFolder, ModelInfo, PresignedUrl, S3Obj


USER = "__DUMMY_TRANSFORMERS_USER__"
PASS = "__DUMMY_TRANSFORMERS_PASS__"
FILES = [
    (
        "nested/Test-{}.txt".format(int(time.time())),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/input.txt"),
    ),
    (
        "nested/yoyo {}.txt".format(int(time.time())),  # space is intentional
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/empty.txt"),
    ),
]
ENDPOINT_STAGING = "https://moon-staging.huggingface.co"


class HfApiCommonTest(unittest.TestCase):
    _api = HfApi(endpoint=ENDPOINT_STAGING)


class HfApiLoginTest(HfApiCommonTest):
    def test_login_invalid(self):
        with self.assertRaises(HTTPError):
            self._api.login(username=USER, password="fake")

    def test_login_valid(self):
        token = self._api.login(username=USER, password=PASS)
        self.assertIsInstance(token, str)


class HfApiEndpointsTest(HfApiCommonTest):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = cls._api.login(username=USER, password=PASS)

    @classmethod
    def tearDownClass(cls):
        for FILE_KEY, FILE_PATH in FILES:
            cls._api.delete_obj(token=cls._token, filename=FILE_KEY)

    def test_whoami(self):
        user, orgs = self._api.whoami(token=self._token)
        self.assertEqual(user, USER)
        self.assertIsInstance(orgs, list)

    def test_presign_invalid_org(self):
        with self.assertRaises(HTTPError):
            _ = self._api.presign(token=self._token, filename="nested/fake_org.txt", organization="fake")

    def test_presign_valid_org(self):
        urls = self._api.presign(token=self._token, filename="nested/valid_org.txt", organization="valid_org")
        self.assertIsInstance(urls, PresignedUrl)

    def test_presign_invalid(self):
        try:
            _ = self._api.presign(token=self._token, filename="non_nested.json")
        except HTTPError as e:
            self.assertIsNotNone(e.response.text)
            self.assertTrue("Filename invalid" in e.response.text)
        else:
            self.fail("Expected an exception")

    def test_presign(self):
        for FILE_KEY, FILE_PATH in FILES:
            urls = self._api.presign(token=self._token, filename=FILE_KEY)
            self.assertIsInstance(urls, PresignedUrl)
            self.assertEqual(urls.type, "text/plain")

    def test_presign_and_upload(self):
        for FILE_KEY, FILE_PATH in FILES:
            access_url = self._api.presign_and_upload(token=self._token, filename=FILE_KEY, filepath=FILE_PATH)
            self.assertIsInstance(access_url, str)
            with open(FILE_PATH, "r") as f:
                body = f.read()
            r = requests.get(access_url)
            self.assertEqual(r.text, body)

    def test_list_objs(self):
        objs = self._api.list_objs(token=self._token)
        self.assertIsInstance(objs, list)
        if len(objs) > 0:
            o = objs[-1]
            self.assertIsInstance(o, S3Obj)


class HfApiPublicTest(unittest.TestCase):
    def test_staging_model_list(self):
        _api = HfApi(endpoint=ENDPOINT_STAGING)
        _ = _api.model_list()

    def test_model_list(self):
        _api = HfApi()
        models = _api.model_list()
        self.assertGreater(len(models), 100)
        self.assertIsInstance(models[0], ModelInfo)


class HfFolderTest(unittest.TestCase):
    def test_token_workflow(self):
        """
        Test the whole token save/get/delete workflow,
        with the desired behavior with respect to non-existent tokens.
        """
        token = "token-{}".format(int(time.time()))
        HfFolder.save_token(token)
        self.assertEqual(HfFolder.get_token(), token)
        HfFolder.delete_token()
        HfFolder.delete_token()
        # ^^ not an error, we test that the
        # second call does not fail.
        self.assertEqual(HfFolder.get_token(), None)
