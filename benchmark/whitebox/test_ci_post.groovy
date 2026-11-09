// Execute the real post block in Jenkins Declarative's documented condition order.
def source = new File(args[0]).text
def postSource = source.substring(source.lastIndexOf('\n    post {')).trim()
postSource = postSource.substring(0, postSource.lastIndexOf('}')).trim()
int cases = 0
['SUCCESS', 'FAILURE', 'ABORTED', 'UNSTABLE'].each { result ->
  def handlers = [:]
  boolean cleaned = false
  def calls = []
  def variables = [
    post: { Closure body -> body() },
    script: { Closure body -> body() },
    container: { String name, Closure body -> body() },
    sh: { Map step ->
      assert step.script.contains('/build-secrets/verified-acr-image.json')
      cleaned = true
      calls << 'cleanup'
      return 0
    },
    callbackPlatform: { String status ->
      assert !cleaned: 'Verified artifact must remain available until callback completes'
      assert status == result
      calls << status
    },
  ]
  // Declarative ignores source order and evaluates always before success.
  def conditionOrder = ['always', 'changed', 'fixed', 'regression', 'aborted',
                        'failure', 'success', 'unstable', 'unsuccessful', 'cleanup']
  conditionOrder.each { name -> variables[name] = { Closure body -> handlers[name] = body } }
  new GroovyShell(new Binding(variables)).evaluate(postSource)
  conditionOrder.each { name ->
    if (name in ['always', 'cleanup', result.toLowerCase()]) handlers[name]?.call()
  }
  assert calls == [result, 'cleanup']
  assert cleaned
  cases++
}
println "$cases post lifecycle cases passed"
