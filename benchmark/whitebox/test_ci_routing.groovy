def source = new File(args[0]).text.split(/(?m)^pipeline \{/)[0]
def sha = '2cb2887bee1d87d8ccac4353eeb56005a3c2c6cd'
def ecr = '829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-realtime'
def bj = 'loopit-registry-bj-registry.cn-beijing.cr.aliyuncs.com/minwm/sglang-minwm-realtime'
def th = 'lwdp-codex-th-registry-vpc.ap-southeast-7.cr.aliyuncs.com/lwdp/whitebox-runtime'
def pub = th.replace('-registry-vpc.', '-registry.')
def run = { job, service, image, branch='main' ->
  def env = [JOB_BASE_NAME:job, ECR_IMAGE_REPOSITORY:ecr, ACR_IMAGE_REPOSITORY:bj, BUILD_NUMBER:'1']
  def binding = new Binding([env:env, params:[BRANCH_NAME:branch, SERVICE_NAME:service, IMAGE:image], currentBuild:[:],
    sh:{ m -> m.script == 'git rev-parse HEAD' ? sha : '' }, error:{ m -> throw new IllegalArgumentException(m.toString()) }])
  def script = new GroovyShell(binding).parse(source)
  script.prepareWorldModelBuild()
  return env
}
int count=0
['white_box','codex/candidate'].each { branch ->
 boolean rejected=false
 try {run('whitebox-th-room','whitebox-th-room',null,branch)} catch(IllegalArgumentException e) {rejected=true}
 assert rejected;count++
}
assert run('whitebox-th-ui','world-test-platform-whitebox',null,'white_box').WHITEBOX_SOURCE_BRANCH == 'white_box'
[['whitebox-th-room','whitebox-th-room','whitebox-room'], ['whitebox-th-control','whitebox-th-gateway','whitebox-control'], ['whitebox-th-control','whitebox-th-coordinator','whitebox-control']].each { row ->
  def out=run(row[0],row[1],null)
  assert out.IMAGE_REF == "$th:${row[2]}-$sha"
  assert out.ACR_IMAGE_REF == "$pub:${row[2]}-$sha"
  assert out.ECR_IMAGE_REF == ''
  assert out.BUILDKIT_IMAGE_OUTPUT == "type=image,name=${out.ACR_IMAGE_REF},push=true"
  assert out.ACR_REGION == 'ap-southeast-7';count++
  [ecr,bj,pub,'attacker.example/runtime'].each { repo ->
    boolean rejected=false
    try {run(row[0],row[1],"$repo:white-box-20261015163000")} catch(IllegalArgumentException e) {rejected=true}
    assert rejected;count++
  }
  out=run(row[0],row[1],"$th:main-012345abcdef-20261015163000")
  assert out.IMAGE_REF == "$th:main-012345abcdef-20261015163000";count++
}
['whitebox-th-ui','whitebox-ui'].each { job ->
 def out=run(job,'world-test-platform-whitebox',"$bj:white-box-20261015163000",'white_box')
 assert out.THAILAND_ACR_ONLY == 'false'
 assert out.ACR_REGION == 'cn-beijing'
 assert out.BUILDKIT_IMAGE_OUTPUT.contains(ecr)
 assert out.BUILDKIT_IMAGE_OUTPUT.contains(bj);count++
}
def out=run('whitebox-control','whitebox-gateway',null)
assert out.IMAGE_REF == "$ecr:whitebox-control-$sha";count++
println "$count routing contract cases passed"
